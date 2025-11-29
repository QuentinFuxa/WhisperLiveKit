#!/usr/bin/env python3
import logging
import platform
import sys
import time
from functools import lru_cache

import librosa
import numpy as np

from whisperlivekit.backend_support import (faster_backend_available,
                                            mlx_backend_available)
from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.warmup import warmup_asr

from .backends import FasterWhisperASR, MLXWhisper, OpenaiApiASR, WhisperASR

logger = logging.getLogger(__name__)


WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(
    ","
)


def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert (
        lan in WHISPER_LANG_CODES
    ), "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk

        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)

        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if (
        lan
        in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split()
    ):
        from mosestokenizer import MosesSentenceSplitter        

        return MosesSentenceSplitter(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if (
        lan
        in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split()
    ):
        logger.debug(
            f"{lan} code is not supported by wtpsplit. Going to use None lang_code option."
        )
        lan = None

    from wtpsplit import WtP

    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")

    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)

    return WtPtok()


def backend_factory(
            backend,
            lan,
            model_size,
            model_cache_dir,
            model_dir,
            model_path,
            lora_path,
            direct_english_translation,
            buffer_trimming,
            buffer_trimming_sec,
            confidence_validation,
            warmup_file=None,
            min_chunk_size=None,
        ):
    backend_choice = backend
    custom_reference = model_path or model_dir
    resolved_root = None
    has_mlx_weights = False
    has_fw_weights = False
    has_pytorch = False

    if custom_reference:
        resolved_root = resolve_model_path(custom_reference)
        if resolved_root.is_dir():
            model_info = detect_model_format(resolved_root)
            has_mlx_weights = model_info.compatible_whisper_mlx
            has_fw_weights = model_info.compatible_faster_whisper
            has_pytorch = model_info.has_pytorch
        else:
            # Single file provided
            has_pytorch = True

    if backend_choice == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=lan)
    else:
        backend_choice = _normalize_backend_choice(
            backend_choice,
            resolved_root,
            has_mlx_weights,
            has_fw_weights,
        )

        if backend_choice == "faster-whisper":
            asr_cls = FasterWhisperASR
            if resolved_root is not None and not resolved_root.is_dir():
                raise ValueError("Faster-Whisper backend expects a directory with CTranslate2 weights.")
            model_override = str(resolved_root) if resolved_root is not None else None
        elif backend_choice == "mlx-whisper":
            asr_cls = MLXWhisper
            if resolved_root is not None and not resolved_root.is_dir():
                raise ValueError("MLX Whisper backend expects a directory containing MLX weights.")
            model_override = str(resolved_root) if resolved_root is not None else None
        else:
            asr_cls = WhisperASR
            model_override = str(resolved_root) if resolved_root is not None else None
            if custom_reference and not has_pytorch:
                raise FileNotFoundError(
                    f"No PyTorch checkpoint found under {resolved_root or custom_reference}"
                )

        t = time.time()
        logger.info(f"Loading Whisper {model_size} model for language {lan} using backend {backend_choice}...")
        asr = asr_cls(
            model_size=model_size,
            lan=lan,
            cache_dir=model_cache_dir,
            model_dir=model_override,
            lora_path=lora_path if backend_choice == "whisper" else None,
        )
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    if direct_english_translation:
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = lan  # Whisper transcribes in this language

    # Create the tokenizer
    if buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None
    
    warmup_asr(asr, warmup_file)
    
    asr.confidence_validation = confidence_validation
    asr.tokenizer = tokenizer
    asr.buffer_trimming = buffer_trimming
    asr.buffer_trimming_sec = buffer_trimming_sec
    asr.backend_choice = backend_choice
    return asr


def _normalize_backend_choice(
    preferred_backend,
    resolved_root,
    has_mlx_weights,
    has_fw_weights,
):
    backend_choice = preferred_backend

    if backend_choice == "auto":
        if mlx_backend_available(warn_on_missing=True) and (resolved_root is None or has_mlx_weights):
            return "mlx-whisper"
        if faster_backend_available(warn_on_missing=True) and (resolved_root is None or has_fw_weights):
            return "faster-whisper"
        return "whisper"

    if backend_choice == "mlx-whisper":
        if not mlx_backend_available():
            raise RuntimeError("mlx-whisper backend requested but mlx-whisper is not installed.")
        if resolved_root is not None and not has_mlx_weights:
            raise FileNotFoundError(
                f"mlx-whisper backend requested but no MLX weights were found under {resolved_root}"
            )
        if platform.system() != "Darwin":
            logger.warning("mlx-whisper backend requested on a non-macOS system; this may fail.")
        return backend_choice

    if backend_choice == "faster-whisper":
        if not faster_backend_available():
            raise RuntimeError("faster-whisper backend requested but faster-whisper is not installed.")
        if resolved_root is not None and not has_fw_weights:
            raise FileNotFoundError(
                f"faster-whisper backend requested but no Faster-Whisper weights were found under {resolved_root}"
            )
        return backend_choice

    if backend_choice == "whisper":
        return backend_choice

    raise ValueError(f"Unknown backend '{preferred_backend}' for LocalAgreement.")

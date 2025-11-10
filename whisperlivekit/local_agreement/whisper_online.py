#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
from .backends import FasterWhisperASR, MLXWhisper, WhisperTimestampedASR, OpenaiApiASR
from whisperlivekit.warmup import warmup_asr

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
            task,
            buffer_trimming,
            buffer_trimming_sec,
            confidence_validation,
            warmup_file=None,
            min_chunk_size=None,
        ):
    backend = backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=lan)        
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR

        t = time.time()
        logger.info(f"Loading Whisper {model_size} model for language {lan}...")
        asr = asr_cls(
            model_size=model_size,
            lan=lan,
            cache_dir=model_cache_dir,
            model_dir=model_dir,
        )
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    if task == "translate":
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
    return asr
"""Model-config profiles for decoder-LLM translation backends.

A profile holds the model-specific values that do not depend on the inference
engine: the prompt template, the EOS token, and the sampling parameters. A
backend (mlx-lm, vLLM, etc.) loads a model from its own repo field and runs
its own decode loop, but shares the profile so the prompt and sampling stay
in one place.

Add a model by adding an entry here, not by writing backend code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MtModelProfile:
    """Model-specific values shared across inference engines.

    ``repo`` is the HF repo for the default engine (mlx-lm). A backend that
    needs a different repo (e.g. vLLM with a non-quantized checkpoint) resolves
    its own repo from the profile name or a per-engine override.

    ``prompt_kind`` selects how the backend builds the chat message:

    * ``"text"`` (default) — ``prompt_template`` formats with
      ``{target_lang}`` (the target language name) and ``{text}``. Some models
      (e.g. Hunyuan-MT) ship two templates: one for ZH⇔XX (Chinese instruction)
      and one for XX⇔XX (English instruction), and require the language NAME to
      match the instruction language — Chinese names in the Chinese template,
      English names in the English template. ``lang_names`` and
      ``lang_names_zh`` provide those maps; the resolver picks the right one
      for the chosen template.
    * ``"structured_chat"`` — the model's chat template handles prompt
      construction internally. The backend passes ISO language codes (e.g.
      ``en``, ``it``, ``zh``) directly in a structured content list;
      ``prompt_template``/``prompt_template_xx``/``lang_names``/
      ``lang_names_zh`` are unused.
    """
    repo: str
    prompt_template: str = ""
    eos_token: Optional[str] = None
    temp: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.1
    max_tokens: int = 512
    #: Template for XX⇔XX (neither side Chinese). When None, ``prompt_template`` is used for all directions.
    prompt_template_xx: Optional[str] = None
    #: Language code → English full name, for the English-instruction template.
    lang_names: Dict[str, str] = field(default_factory=dict)
    #: Language code → Chinese full name, for the Chinese-instruction template.
    lang_names_zh: Dict[str, str] = field(default_factory=dict)
    #: How the backend builds the prompt: "text" (template.format) or "structured_chat" (ISO codes in structured content).
    prompt_kind: str = "text"


# The registry. Backends read from this; profiles populate it at import time.
MT_MODEL_PROFILES: Dict[str, MtModelProfile] = {}


# Language codes that count as "Chinese" for the ZH⇔XX template rule.
_ZH_CODES = {"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "cmn", "yue"}


def resolve_prompt(profile: MtModelProfile, source_language: str, target_language: str) -> dict:
    """Resolve the prompt specification for a source→target pair.

    Returns a dict the backend branches on:

    * ``{"kind": "text", "template": ..., "target_name": ...}`` — the
      backend formats ``template`` with ``target_lang=target_name`` and
      ``text=<source>`` to produce a string content.
    * ``{"kind": "structured_chat", "src": <iso>, "tgt": <iso>}`` — the
      backend builds a structured content list with the ISO codes directly;
      the model's chat template handles the rest.
    """
    src = (source_language or "").strip().lower()
    tgt = (target_language or "").strip().lower()

    if profile.prompt_kind == "structured_chat":
        # Normalize Chinese variants to the base ISO code.
        if src in _ZH_CODES:
            src = "zh"
        if tgt in _ZH_CODES:
            tgt = "zh"
        return {"kind": "structured_chat", "src": src, "tgt": tgt}

    # Text kind: pick template + language name.
    either_zh = src in _ZH_CODES or tgt in _ZH_CODES
    if not either_zh and profile.prompt_template_xx is not None:
        template = profile.prompt_template_xx
        names = profile.lang_names
    else:
        template = profile.prompt_template
        names = profile.lang_names_zh or profile.lang_names
    target_name = names.get(tgt, target_language)
    return {"kind": "text", "template": template, "target_name": target_name}


# ---------------------------------------------------------------------------
# Hunyuan-MT family
# ---------------------------------------------------------------------------
# Two prompt templates from the official repo (github.com/Tencent-Hunyuan/Hy-MT):
# ZH⇔XX uses the Chinese instruction; XX⇔XX uses the English instruction.
_HUNYUAN_PROMPT_ZH = "把下面的文本翻译成{target_lang}，不要额外解释。\n\n{text}"
_HUNYUAN_PROMPT_XX = "Translate the following segment into {target_lang}, without additional explanation.\n\n{text}"
_HUNYUAN_EOS = "<|im_end|>"

# Language names for the Chinese-instruction template (full Chinese names).
# From the official Hunyuan-MT language table.
_HUNYUAN_NAMES_ZH = {
    "zh": "中文", "zh-cn": "简体中文", "zh-tw": "繁體中文", "zh-hans": "简体中文", "zh-hant": "繁體中文",
    "en": "英语", "fr": "法语", "es": "西班牙语", "ja": "日语", "ko": "韩语",
    "de": "德语", "it": "意大利语", "pt": "葡萄牙语", "ru": "俄语", "ar": "阿拉伯语",
    "th": "泰语", "tr": "土耳其语", "id": "印尼语", "ms": "马来语", "vi": "越南语",
}
# Language names for the English-instruction template (full English names).
_HUNYUAN_NAMES_EN = {
    "zh": "Chinese", "zh-cn": "Simplified Chinese", "zh-tw": "Traditional Chinese",
    "zh-hans": "Simplified Chinese", "zh-hant": "Traditional Chinese",
    "en": "English", "fr": "French", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "th": "Thai", "tr": "Turkish", "id": "Indonesian", "ms": "Malay", "vi": "Vietnamese",
}

_HUNYUAN_MODELS = {
    "hy-mt2-1.8b-8bit": "mlx-community/Hy-MT2-1.8B-8bit",
    "hy-mt2-1.8b-4bit": "mlx-community/Hy-MT2-1.8B-4bit",
    "hy-mt2-7b-4bit": "mlx-community/Hy-MT2-7B-4bit",
    "hy-mt2-7b-8bit": "mlx-community/Hy-MT2-7B-8bit",
    "hunyuan-mt-7b-4bit": "mlx-community/Hunyuan-MT-7B-4bit",
    "hunyuan-mt-7b-8bit": "mlx-community/Hunyuan-MT-7B-8bit",
}

for _name, _repo in _HUNYUAN_MODELS.items():
    MT_MODEL_PROFILES[_name] = MtModelProfile(
        repo=_repo,
        prompt_template=_HUNYUAN_PROMPT_ZH,
        prompt_template_xx=_HUNYUAN_PROMPT_XX,
        eos_token=_HUNYUAN_EOS,
        lang_names=_HUNYUAN_NAMES_EN,
        lang_names_zh=_HUNYUAN_NAMES_ZH,
    )

# ---------------------------------------------------------------------------
# TranslateGemma family (multilingual, including en→it, zh→en)
# ---------------------------------------------------------------------------
# TranslateGemma uses a STRUCTURED chat template, not a text prompt. The
# chat template maps ISO codes to full names internally, so no name map is
# needed. Source/target are ISO codes (en, it, zh), not full names.
# prompt_template/prompt_template_xx/lang_names are unused for structured_chat.

MT_MODEL_PROFILES["translategemma-4b-it-4bit"] = MtModelProfile(
    repo="mlx-community/translategemma-4b-it-4bit",
    prompt_kind="structured_chat",
    # TranslateGemma's tokenizer eos_token is <eos>, but the model generates
    # <end_of_turn> as its turn-end marker. The tokenizer's eos_token_ids does
    # not include <end_of_turn>, so mlx-lm's stream_generate won't stop on it.
    # Set it explicitly so the manual EOS check in the backend breaks the loop.
    eos_token="<end_of_turn>",
)

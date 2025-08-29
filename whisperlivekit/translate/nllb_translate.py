# nllb_translate.py
import argparse
from pathlib import Path
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "facebook/nllb-200-distilled-600M"

# Common language shortcuts → NLLB codes (extend as needed)
LANG_MAP = {
    "english": "eng_Latn",
    "en": "eng_Latn",
    "french": "fra_Latn",
    "fr": "fra_Latn",
    "spanish": "spa_Latn",
    "es": "spa_Latn",
    "german": "deu_Latn",
    "de": "deu_Latn",
    "italian": "ita_Latn",
    "it": "ita_Latn",
    "portuguese": "por_Latn",
    "pt": "por_Latn",
    "arabic": "arb_Arab",
    "ar": "arb_Arab",
    "russian": "rus_Cyrl",
    "ru": "rus_Cyrl",
    "turkish": "tur_Latn",
    "tr": "tur_Latn",
    "chinese": "zho_Hans",
    "zh": "zho_Hans",           # Simplified
    "zh-cn": "zho_Hans",
    "zh-hans": "zho_Hans",
    "zh-hant": "zho_Hant",      # Traditional
    "japanese": "jpn_Jpan",
    "ja": "jpn_Jpan",
    "korean": "kor_Hang",
    "ko": "kor_Hang",
    "dutch": "nld_Latn",
    "nl": "nld_Latn",
    "polish": "pol_Latn",
    "pl": "pol_Latn",
    "swedish": "swe_Latn",
    "sv": "swe_Latn",
    "norwegian": "nob_Latn",
    "no": "nob_Latn",
    "danish": "dan_Latn",
    "da": "dan_Latn",
    "finnish": "fin_Latn",
    "fi": "fin_Latn",
    "catalan": "cat_Latn",
    "ca": "cat_Latn",
    "hindi": "hin_Deva",
    "hi": "hin_Deva",
    "vietnamese": "vie_Latn",
    "vi": "vie_Latn",
    "indonesian": "ind_Latn",
    "id": "ind_Latn",
    "thai": "tha_Thai",
    "th": "tha_Thai",
}

def norm_lang(code: str) -> str:
    c = code.strip().lower()
    return LANG_MAP.get(c, code)

def translate_texts(texts: List[str], src_code: str, tgt_code: str,
                    max_new_tokens=512, device=None, dtype=None) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=src_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype if dtype is not None else (torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if device:
        model.to(device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    if device or torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            forced_bos_token_id=forced_bos,
        )
    outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [o.strip() for o in outs]

def main():
    ap = argparse.ArgumentParser(description="Translate with facebook/nllb-200-distilled-600M")
    ap.add_argument("--text", help="Inline text to translate")
    ap.add_argument("--file", help="Path to a UTF-8 text file (one example per line)")
    ap.add_argument("--src", required=True, help="Source language (e.g. fr, fra_Latn)")
    ap.add_argument("--tgt", required=True, help="Target language (e.g. en, eng_Latn)")
    ap.add_argument("--max-new", type=int, default=512, help="Max new tokens")
    args = ap.parse_args()

    src = norm_lang(args.src)
    tgt = norm_lang(args.tgt)

    batch: List[str] = []
    if args.text:
        batch.append(args.text)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        batch.extend([ln for ln in lines if ln.strip()])

    if not batch:
        raise SystemExit("Provide --text or --file")

    results = translate_texts(batch, src, tgt, max_new_tokens=args.max_new)
    for i, (inp, out) in enumerate(zip(batch, results), 1):
        print(f"\n--- Sample {i} ---")
        print(f"SRC [{src}]: {inp}")
        print(f"TGT [{tgt}]: {out}")

if __name__ == "__main__":
    main()

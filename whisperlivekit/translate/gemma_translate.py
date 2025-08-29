# gemma_translate.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-270m-it"

def build_prompt(tokenizer, text, target_lang, source_lang=None):
    # Use the model's chat template for best results
    if source_lang:
        user_msg = (
            f"Translate the following {source_lang} text into {target_lang}.\n"
            f"Return only the translation.\n\n"
            f"Text:\n{text}"
        )
    else:
        user_msg = (
            f"Translate the following text into {target_lang}.\n"
            f"Return only the translation.\n\n"
            f"Text:\n{text}"
        )
    chat = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def translate(text, target_lang, source_lang=None, max_new_tokens=256, temperature=0.2, top_p=0.95):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    prompt = build_prompt(tokenizer, text, target_lang, source_lang)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0.0,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt to keep only the assistant answer
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Translate with google/gemma-3-270m-it")
    ap.add_argument("--text", required=True, help="Text to translate")
    ap.add_argument("--to", dest="target_lang", required=True, help="Target language (e.g., French, Spanish)")
    ap.add_argument("--from", dest="source_lang", default=None, help="Source language (optional)")
    ap.add_argument("--temp", type=float, default=0.2, help="Sampling temperature (0 = deterministic-ish)")
    ap.add_argument("--max-new", type=int, default=256, help="Max new tokens")
    args = ap.parse_args()

    print(translate(args.text, args.target_lang, args.source_lang, max_new_tokens=args.max_new, temperature=args.temp))

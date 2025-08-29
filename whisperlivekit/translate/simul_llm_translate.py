import sys

import ctranslate2
import sentencepiece as spm
import transformers
import argparse

def generate_words(sp, step_results):
    tokens_buffer = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and tokens_buffer:
            word = sp.decode(tokens_buffer)
            if word:
                yield word
            tokens_buffer = []

        tokens_buffer.append(step_result.token_id)

    if tokens_buffer:
        word = sp.decode(tokens_buffer)
        if word:
            yield word

from sentence_segmenter import SentenceSegmenter

class LLMTranslator:

    def __init__(self, system_prompt='Please translate.', max_context_length=4096, len_ratio=None):
        self.system_prompt = system_prompt


        print("Loading the model...", file=sys.stderr)
        self.generator = ctranslate2.Generator("ct2_EuroLLM-9B-Instruct/", device="cuda")
        self.sp = spm.SentencePieceProcessor("EuroLLM-9B-Instruct/tokenizer.model")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EuroLLM-9B-Instruct/")
        print("...done", file=sys.stderr)

        self.max_context_length = max_context_length

        self.max_tokens_to_trim = self.max_context_length - 10
        self.len_ratio = len_ratio

        # my regex sentence segmenter
        self.segmenter = SentenceSegmenter()

#        self.max_generation_length = 512
#        self.max_prompt_length = context_length - max_generation_length

    def start_dialog(self):
        return [{'role':'system', 'content': self.system_prompt }]
    

    def build_prompt(self, dialog):
        toks = self.tokenizer.apply_chat_template(dialog, tokenize=True, add_generation_prompt=False)
        if len(dialog) == 3:
            toks = toks[:-2]
        print("len toks:", len(toks), file=sys.stderr)
#        print(toks, file=sys.stderr)

        c = self.tokenizer.convert_ids_to_tokens(toks)
#        print(c,file=sys.stderr)
        return c

    def translate(self, src, tgt_forced=""):
        #src, tgt_forced = self.trim(src, tgt_forced)

        dialog = self.start_dialog()
        dialog += [{'role':'user','content': src}]
        if tgt_forced != "":
            dialog += [{'role':'assistant','content': tgt_forced}]

        prompt_tokens = self.build_prompt(dialog)
        if self.len_ratio is not None:
            limit_len = int(len(self.tokenizer.encode(src)) * self.len_ratio) + 10
            limit_kw = {'max_length': limit_len}
        else:
            limit_kw = {}
        step_results = self.generator.generate_tokens(
            prompt_tokens,
            **limit_kw, 
    #    end_token=tokenizer.eos_token,
    #            sampling_temperature=0.6,
    #            sampling_topk=20,
    #            sampling_topp=1,
        )

        res = []
        #output_ids = []
        for step_result in step_results:
        #    is_new_word = step_result.token.startswith("▁")
        #    if is_new_word and output_ids:
        #        word = self.sp.decode(output_ids)
#                print(word, end=" ", flush=True, file=sys.stderr)
        #        output_ids = []
        #    output_ids.append(step_result.token_id)
            res.append(step_result)

        #if output_ids:
        #    word = self.sp.decode(output_ids)
#        print(word, file=sys.stderr)

        return self.sp.decode([r.token_id for r in res])
 #       print(res)
 #       print([s.token for s in res], file=sys.stderr)
#        print([s.token==self.tokenizer.eos_token for s in res], file=sys.stderr)

class ParallelTextBuffer:
    def __init__(self, tokenizer, max_tokens, trimming="segments", init_src="", init_tgt=""):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.src_buffer = []  # list of lists
        if init_src:
            self.src_buffer.append(init_src)

        self.tgt_buffer = []  # list of strings
        if init_tgt:
            self.tgt_buffer.append(init_tgt)

        self.trimming = trimming
        if self.trimming == "sentences":
            self.segmenter = SentenceSegmenter()

    def len_src(self):
        return sum(len(t) for t in self.src_buffer) + len(self.src_buffer) - 1

    def insert(self, src, tgt):
        self.src_buffer.append(src)
        self.tgt_buffer.append(tgt)

    def insert_src_suffix(self, s):
        if self.src_buffer:
            self.src_buffer[-1][-1] += s
        else:
            self.src_buffer.append([s])

    def trim_sentences(self):
        # src_tok_lens = [len(self.tokenizer.encode(" ".join(b))) for b in self.src_buffer]
        # tgt_tok_lens = [len(self.tokenizer.encode(t)) for t in self.tgt_buffer]

        src = " ".join(" ".join(b) for b in self.src_buffer)
        tgt = "".join(self.tgt_buffer)

        src_sp_toks = self.tokenizer.encode(src)
        tgt_sp_toks = self.tokenizer.encode(tgt)



        def trim_sentence(text):
            sents = self.segmenter(text)
            print("SENTS:", len(sents), sents, file=sys.stderr)
            return "".join(sents[1:])

        while len(src_sp_toks) + len(tgt_sp_toks) > self.max_tokens:
            nsrc = trim_sentence(src)
            ntgt = trim_sentence(tgt)
            if not nsrc or not ntgt:
                print("src or tgt is empty after trimming.", file=sys.stderr)
                print("src: ", src, file=sys.stderr)
                print("tgt: ", tgt, file=sys.stderr)
                break
            src = nsrc
            tgt = ntgt
            src_sp_toks = self.tokenizer.encode(src)
            tgt_sp_toks = self.tokenizer.encode(tgt)
            print("TRIMMED SRC:", (src,), file=sys.stderr)
            print("TRIMMED TGT:", (tgt,), file=sys.stderr)

        self.src_buffer = [src.split()]
        self.tgt_buffer = [tgt]
        return src, tgt

    def trim_segments(self):
        print("BUFFER:", file=sys.stderr)
        for s,t in zip(self.src_buffer, self.tgt_buffer):
            print("\t", s,"...",t,file=sys.stderr) #,self.src_buffer, self.tgt_buffer, file=sys.stderr)
        src = " ".join(" ".join(b) for b in self.src_buffer)
        tgt = "".join(self.tgt_buffer)

        src_sp_toks = self.tokenizer.encode(src)
        tgt_sp_toks = self.tokenizer.encode(tgt)

        while len(src_sp_toks) + len(tgt_sp_toks) > self.max_tokens:
            if len(self.src_buffer) > 1 and len(self.tgt_buffer) > 1:
                self.src_buffer.pop(0)
                self.tgt_buffer.pop(0)
            else:
                break
            src = " ".join(" ".join(b) for b in self.src_buffer)
            tgt = "".join(self.tgt_buffer)

            src_sp_toks = self.tokenizer.encode(src)
            tgt_sp_toks = self.tokenizer.encode(tgt)

        print("TRIMMED SEGMENTS SRC:", (src,), file=sys.stderr)
        print("TRIMMED SEGMENTS TGT:", (tgt,), file=sys.stderr)

        return src, tgt

    def trim(self):
        if self.trimming == "sentences":
            return self.trim_sentences()
        return self.trim_segments()



class SimulLLM:

    def __init__(self, llmtrans, min_len=0, chunk=1, trimming="sentences", language="ja", init_src="", init_tgt=""):
        self.llmtranslator = llmtrans

        #self.src_buffer = init_src 
        #self.confirmed_tgt = init_tgt

        self.buffer = ParallelTextBuffer(self.llmtranslator.tokenizer, self.llmtranslator.max_tokens_to_trim, trimming=trimming, init_src=init_src, init_tgt=init_tgt)

        self.last_inserted = []
        self.last_unconfirmed = ""

        self.min_len = min_len

        self.step = chunk 
        self.language = language
        if language in ["ja", "zh"]:
            self.specific_space = ""
        else:
            self.specific_space = " "

    def insert(self, src):
        if isinstance(src, str):
            self.last_inserted.append(src)
        else:
            self.last_inserted += src

    def insert_suffix(self, text):
        '''
        Insert suffix of a word to the last inserted word. 
        It may be because the word was split to multiple parts in the input, each with different timestamps.
        '''
        if self.last_inserted:
            self.last_inserted[-1] += text
        elif self.src_buffer:
            self.buffer.insert_src_suffix(text)
        else:
            # this shouldn't happen
            self.last_inserted.append(text)

    def trim_longest_common_prefix(self, a,b):
        if self.language not in ["ja", "zh"]:
            a = a.split()
            b = b.split()
        i = 0
        for i,(x,y) in enumerate(zip(a,b)):
            if x != y:
                break
        if self.language in ["ja", "zh"]:
            #print("tady160",(a, b, i), file=sys.stderr)
            return a[:i], b[i:]
        else:
            return " ".join(a[:i]), " ".join(b[i:])

    def process_iter(self):
        if self.buffer.len_src() + len(self.last_inserted) < self.min_len:
            return ""

        src, forced_tgt = self.buffer.trim() #llmtranslator.trim(" ".join(self.src_buffer), self.confirmed_tgt)
        #self.src_buffer = self.src_buffer.split()
        #src = " ".join(self.src_buffer)

        confirmed_out = ""
        run = False
        for i in range(self.step, len(self.last_inserted), self.step):
            for w in self.last_inserted[i-self.step:i]:
                src += " " + w
                run = True
            if not run: break
            
            print("SRC",src,file=sys.stderr)

            print("FORCED TGT",forced_tgt,file=sys.stderr)
            out = self.llmtranslator.translate(src, forced_tgt)
            print("OUT",out,file=sys.stderr)
            confirmed, unconfirmed = self.trim_longest_common_prefix(self.last_unconfirmed, out)
            self.last_unconfirmed = unconfirmed
            #print("tady", (self.confirmed_tgt, self.specific_space, confirmed), file=sys.stderr)
            if confirmed:
#                self.confirmed_tgt += self.specific_space + confirmed
#            print(confirmed_out, confirmed, file=sys.stderr)
                confirmed_out += self.specific_space + confirmed
            print("CONFIRMED NOW:",confirmed,file=sys.stderr)


            print(file=sys.stderr)
            print(file=sys.stderr)
        print("#################",file=sys.stderr)
        if run:
            self.buffer.insert(self.last_inserted, confirmed_out)
            self.last_inserted = []

        ret = confirmed_out
        print("RET:",ret,file=sys.stderr)
        return ret

    def finalize(self):
        return self.last_unconfirmed


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input-instance', type=str, default=None, help="Filename of instances to simulate input. If not set, txt input is read from stdin.")
#parser.add_argument('--output_instance', type=str, default=None, help="Write output as instance into this file, while also writing to stdout.")
parser.add_argument('--min-chunk-size', type=int, default=1, 
                    help='Minimum number of space-delimited words to process in each LocalAgreement update. The more, the higher quality, but slower.')
parser.add_argument('--min-len', type=int, default=1, 
                    help='Minimum number of space-delimited words at the beginning.')
#parser.add_argument('--start_at', type=int, default=0, help='Skip first N words.')

# maybe later
#parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
#parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')

lan_to_name = {
    "de": "German",
    "ja": "Japanese",
    "zh-tr": "Chinese Traditional",
    "zh-sim": "Chinese Simplified",
    "cs": "Czech",
    }
parser.add_argument('--lan', '--language', type=str, default="de", 
                    help="Target language code.",
                    choices=["de", "ja","zh-tr","zh-sim","cs"])

SrcLang = "English"  # always
TgtLang = "German"
default_prompt="You are simultaneous interpreter from {SrcLang} to {TgtLang}. We are at a conference. It is important that you translate " + \
                "only what you hear, nothing else!"
parser.add_argument('--sys_prompt', type=str, default=None, 
                    help='System prompt. If None, default one is used, depending on the language. The prompt should ')

default_init = "Please, go ahead, you can start with your presentation, we are ready."


default_inits_tgt = {
    'de': "Bitte schön, Sie können mit Ihrer Präsentation beginnen, wir sind bereit.",
    'ja': "どうぞ、プレゼンテーションを始めてください。",  # # Please go ahead and start your presentation.  # this is in English
    'zh-tr': "請繼續，您可以開始您的簡報，我們已經準備好了。",
    'zh-sim': "请吧，你可以开始发言了，我们已经准备好了。",
    'cs': "Prosím, můžete začít s prezentací, jsme připraveni.",
}
parser.add_argument('--init_prompt_src', type=str, default=None, help='Init translation with source text. It should be a complete sentence in the source language. ' 
                    'It can be context specific for the given input. Default is ')
parser.add_argument('--init_prompt_tgt', type=str, default=None, help='Init translation with this target. It should be example translation of init_prompt_src. '
                    ' There is default init message, depending on the language.')

parser.add_argument('--len-threshold', type=float, default=None, help='Ratio of the length of the source and generated target, in number of sentencepiece tokens. '
                    'It should reflect the target language and. If not set, no len-threshold is used.')

# how many times is target text longer than English
lan_thresholds = {
    'de': 1.3,   # 12751/9817  ... the proportion of subword tokens for ACL6060 dev de vs. en text, for EuroLLM-9B-Instruct tokenizer
    'ja': 1.34,  # 13187/9817
    'zh': 1.23,  # 12115/9817
    'zh-tr': 1.23, # 12115/9817
    'zh-sim': 1.23, # 12115/9817
#    'cs': I don't know    # guessed
}
parser.add_argument('--language-specific-len-threshold', default=False, action="store_true", 
                    help='Use language-specific length threshold, e.g. 1.3 for German.')

parser.add_argument("--max-context-length", type=int, default=4096, help="Maximum number of tokens in the model to use.")

parser.add_argument("--buffer_trimming", type=str, default="sentences", choices=["segments","sentences"], help="Buffer trimming strategy.")

args = parser.parse_args()

if args.sys_prompt is None:
    TgtLang = lan_to_name[args.lan]
    sys_prompt = default_prompt.format(SrcLang=SrcLang, TgtLang=TgtLang)
else:
    sys_prompt = args.sys_prompt

if args.init_prompt_src is None:
    init_src = default_init.split()
    if args.init_prompt_tgt is None:
        init_tgt = default_inits_tgt[args.lan]
        if args.lan == "ja":
            init_src = 'Please go ahead and start your presentation.'.split()
            print("WARNING: Default init_prompt_src not set and language is Japanese. The init_src prompt changed to be more verbose.", file=sys.stderr)
    else:
        print("WARNING: init_prompt_tgt is used, init_prompt_src is None, the default one. It may be wrong!", file=sys.stderr)
        init_tgt = args.init_prompt_tgt
else:
    init_src = args.init_prompt_src.split()
    if args.init_prompt_tgt is None:
        print("WARNING: init_prompt_src is used, init_prompt_tgt is None, so the default one is used. It may be wrong!", file=sys.stderr)
        init_tgt = default_inits_tgt[args.lan]
    else:
        init_tgt = args.init_prompt_tgt

print("INFO: System prompt:", sys_prompt, file=sys.stderr)
print("INFO: Init prompt src:", init_src, file=sys.stderr)
print("INFO: Init prompt tgt:", init_tgt, file=sys.stderr)

if args.language_specific_len_threshold:
    if args.len_threshold is not None:
        print("ERROR: --len-threshold is set, but --language-specific-len-threshold is also set. Only one can be used.", file=sys.stderr)
        sys.exit(1)
    else:
        len_threshold = lan_thresholds[args.lan]
else:
    len_threshold = args.len_threshold

llmtrans = LLMTranslator(system_prompt=sys_prompt, max_context_length=args.max_context_length, len_ratio=len_threshold)
lan = args.lan if not args.lan.startswith("zh") else "zh"
simul = SimulLLM(llmtrans,language=lan, min_len=args.min_len, chunk=args.min_chunk_size,
                init_src=init_src, init_tgt=init_tgt, trimming=args.buffer_trimming
                )

# two input options
if args.input_instance is not None:
    print("INFO: Reading input from file", args.input_instance, file=sys.stderr)
    import json
    with open(args.input_instance, "r") as f:
        instance = json.load(f)

    asr_source = instance["prediction"]
    timestamps = instance["delays"]
    elapsed = instance["elapsed"]

    yield_ts_words = zip(timestamps, timestamps, elapsed, asr_source.split())
else:
    print("INFO: Reading stdin in txt format", file=sys.stderr)
    def yield_input():
        for line in sys.stdin:
            line = line.strip()
            ts, beg, end, *_ = line.split()
            text = line[len(ts)+len(beg)+len(end)+3:]
            ts = float(ts)
            # in rare cases, the first word is a suffix of the previous word, that was split to multiple parts
            if text[0] != " ":
                first, *words = text.split()
                yield (ts, beg, end, " "+first)  # marking the first word with " ", so that it can be later detected and inserted as suffix
            else:
                words = text.split()
            for w in words:
                yield (ts, beg, end, w)
    yield_ts_words = yield_input()

#i = 0
for t,b,e,w in yield_ts_words:
    if w.startswith(" "):  # it is suffix of the previous word
        w = w[1:] 
        simul.insert_suffix(w)
        continue
    simul.insert(w)
    out = simul.process_iter()
    if out:
        print(t,b,e,out,flush=True)
    # if i > 50:
    #     break
#    i += 1
out = simul.finalize()
print(t,b,e,out,flush=True)
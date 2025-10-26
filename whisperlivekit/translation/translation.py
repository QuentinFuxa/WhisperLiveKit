import logging
import time
import ctranslate2
import torch
import transformers
from dataclasses import dataclass, field
import huggingface_hub
from whisperlivekit.translation.mapping_languages import get_nllb_code
from whisperlivekit.timed_objects import Translation

logger = logging.getLogger(__name__)

#In diarization case, we may want to translate just one speaker, or at least start the sentences there

MIN_SILENCE_DURATION_DEL_BUFFER = 3 #After a silence of x seconds, we consider the model should not use the buffer, even if the previous
# sentence is not finished.

@dataclass
class TranslationModel():
    translator: ctranslate2.Translator
    device: str
    tokenizer: dict = field(default_factory=dict)
    backend_type: str = 'ctranslate2'
    nllb_size: str = '600M'
    
    def get_tokenizer(self, input_lang):
        if not self.tokenizer.get(input_lang, False):
            self.tokenizer[input_lang] = transformers.AutoTokenizer.from_pretrained(
                f"facebook/nllb-200-distilled-{self.nllb_size}",
                src_lang=input_lang,
                clean_up_tokenization_spaces=True
            )
        return self.tokenizer[input_lang]
            

def load_model(src_langs, nllb_backend='ctranslate2', nllb_size='600M'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if nllb_backend=='ctranslate2':
        model = f'nllb-200-distilled-{nllb_size}-ctranslate2'
        MODEL_GUY = 'entai2965'
        huggingface_hub.snapshot_download(MODEL_GUY + '/' + model,local_dir=model)
        translator = ctranslate2.Translator(model,device=device)
    elif nllb_backend=='transformers':
        model = f"facebook/nllb-200-distilled-{nllb_size}"
        translator = transformers.AutoModelForSeq2SeqLM.from_pretrained(model)
    tokenizer = dict()
    for src_lang in src_langs:
        if src_lang != 'auto':
            tokenizer[src_lang] = transformers.AutoTokenizer.from_pretrained(model, src_lang=src_lang, clean_up_tokenization_spaces=True)

    translation_model = TranslationModel(
        translator=translator,
        tokenizer=tokenizer,
        backend_type=nllb_backend,
        device = device,
        nllb_size = nllb_size
    )
    for src_lang in src_langs:
        if src_lang != 'auto':
            translation_model.get_tokenizer(src_lang)
    return translation_model

class OnlineTranslation:
    def __init__(self, translation_model: TranslationModel, input_languages: list, output_languages: list):
        self.input_buffer = []
        self.len_processed_buffer = 0
        self.translation_remaining = Translation()
        self.validated = []
        self.translation_pending_validation = ''
        self.translation_model = translation_model
        self.input_languages = input_languages
        self.output_languages = output_languages

    def compute_common_prefix(self, results):
        #we dont want want to prune the result for the moment. 
        if not self.input_buffer:
            self.input_buffer = results
        else:
            for i in range(min(len(self.input_buffer), len(results))):
                if self.input_buffer[i] != results[i]:
                    self.commited.extend(self.input_buffer[:i])
                    self.input_buffer = results[i:]

    def translate(self, input, input_lang, output_lang):
        if not input:
            return ""
        nllb_output_lang = get_nllb_code(output_lang)
            
        tokenizer = self.translation_model.get_tokenizer(input_lang)
        tokenizer_output = tokenizer(input, return_tensors="pt").to(self.translation_model.device)
        
        if self.translation_model.backend_type == 'ctranslate2':
            source = tokenizer.convert_ids_to_tokens(tokenizer_output['input_ids'][0])    
            results = self.translation_model.translator.translate_batch([source], target_prefix=[[nllb_output_lang]])
            target = results[0].hypotheses[0][1:]
            result = tokenizer.decode(tokenizer.convert_tokens_to_ids(target))
        else:
            translated_tokens = self.translation_model.translator.generate(**tokenizer_output, forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_output_lang))
            result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return result
    
    def translate_tokens(self, tokens):
        if tokens:
            text = ' '.join([token.text for token in tokens])
            start = tokens[0].start
            end = tokens[-1].end
            if self.input_languages[0] == 'auto':
                input_lang = tokens[0].detected_language
            else:
                input_lang = self.input_languages[0]
                
            translated_text = self.translate(text,
                                            input_lang,
                                            self.output_languages[0]
                                            )
            translation = Translation(
                text=translated_text,
                start=start,
                end=end,
            )
            return translation
        return None
            

    def insert_tokens(self, tokens):
        self.input_buffer.extend(tokens)
        pass
    
    def process(self):
        i = 0
        if len(self.input_buffer) < self.len_processed_buffer + 3: #nothing new to process
            return self.validated + [self.translation_remaining]
        while i < len(self.input_buffer):
            if self.input_buffer[i].is_punctuation():
                translation_sentence = self.translate_tokens(self.input_buffer[:i+1])
                self.validated.append(translation_sentence)
                self.input_buffer = self.input_buffer[i+1:]
                i = 0
            else:
                i+=1
        self.translation_remaining = self.translate_tokens(self.input_buffer)
        self.len_processed_buffer = len(self.input_buffer)
        return self.validated, [self.translation_remaining]

    def insert_silence(self, silence_duration: float):
        if silence_duration >= MIN_SILENCE_DURATION_DEL_BUFFER:
            self.input_buffer = []
            self.validated += [self.translation_remaining]

if __name__ == '__main__':
    output_lang = 'fr'
    input_lang = "en"
    
    
    test_string = """
    Transcription technology has improved so much in the past few years. Have you noticed how accurate real-time speech-to-text is now?
    """
    test = test_string.split(' ')
    step = len(test) // 3
    
    shared_model = load_model([input_lang], nllb_backend='transformers')
    online_translation = OnlineTranslation(shared_model, input_languages=[input_lang], output_languages=[output_lang])
    
    beg_inference = time.time()    
    for id in range(5):
        val = test[id*step : (id+1)*step]
        val_str = ' '.join(val)
        result = online_translation.translate(val_str, input_lang = input_lang, output_lang = output_lang)
        print(result)
    print('inference time:', time.time() - beg_inference)
import regex
from functools import lru_cache
class SentenceSegmenter:

    """
    Regex sentence splitter for Latin languages, Japanese and Chinese.
    It is based on sacrebleu TokenizerV14International(BaseTokenizer).
    
    Returns: a list of strings, where each string is a sentence.
    Spaces following punctuation are appended after punctuation within the sequence.
    Total number of characters in the output is the same as in the input.  
    """

    sep = 'ŽžŽžSentenceSeparatorŽžŽž'  # string that certainly won't be in src or target
    latin_terminals = '!?.'
    jap_zh_terminals = '。！？'
    terminals = latin_terminals + jap_zh_terminals

    def __init__(self):
        # end of sentence characters:
        terminals = self.terminals
        self._re = [
            # Separate out punctuations preceeded by a non-digit. 
            # If followed by space-like sequence of characters, they are 
            # appended to the punctuation, not to the next sequence.
            (regex.compile(r'(\P{N})(['+terminals+r'])(\p{Z}*)'), r'\1\2\3'+self.sep),
            # Separate out punctuations followed by a non-digit
            (regex.compile(r'('+terminals+r')(\P{N})'), r'\1'+self.sep+r'\2'),
#            # Separate out symbols
            # -> no, we don't tokenize but segment the punctuation
#            (regex.compile(r'(\p{S})'), r' \1 '),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)
        return [ t for t in line.split(self.sep) if t != '' ]
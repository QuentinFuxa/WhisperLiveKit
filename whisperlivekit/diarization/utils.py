import re


def extract_number(s: str) -> int:
    """Extract the first integer from a string, e.g. 'speaker_2' -> 2."""
    m = re.search(r'\d+', s)
    return int(m.group()) if m else 0

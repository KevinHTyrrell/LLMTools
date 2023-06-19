import re

def clean_string(to_clean: str):
    to_clean = to_clean.replace('\n', ' ')
    to_clean = to_clean.replace('\t', ' ')
    to_clean = re.sub('\s+', ' ', to_clean)
    return to_clean

import pandas as pd
import json
import yaml
from pypdf import PdfReader
from misc.string_fns import clean_string


def load_pdf(filepath: str, split_char: str):
    reader = PdfReader(filepath)
    page_content_list = []
    for page in reader.pages:
        page_to_parse = page.extract_text()
        page_clean = clean_string(page_to_parse)
        sentence_list = page_clean.split(split_char)
        page_content_list += sentence_list
    return page_content_list


def read_file(filepath: str):
    if '.json' in filepath:
        file_contents = read_json(filepath)
    elif '.yml' in filepath or '.yaml' in filepath:
        with open(filepath, 'r') as file_reader:
            raw_contents = yaml.safe_load(file_reader)
        file_contents = raw_contents
    elif '.csv' in filepath:
        file_contents = pd.read_csv(filepath)
    else:
        with open(filepath, 'r') as file_reader:
            raw_data = file_reader.read()
        file_contents = raw_data
    return file_contents


def read_json(filepath: str):
    gzip_str = b"\x1f\x8b\x08"
    with open(filepath, 'rb') as file_reader:
        file_start = file_reader.read(len(gzip_str))

    is_gzip = gzip_str == file_start
    if is_gzip:
        with gzip.open(filepath, 'r') as file_reader:
            raw_data = file_reader.read()
    else:
        with open(filepath, 'r') as file_reader:
            raw_data = file_reader.read()
    raw_json = json.loads(raw_data)
    return raw_json


def read_yaml(filepath: str):
    with open(filepath, 'r') as file_reader:
        raw_yaml = yaml.safe_load(file_reader)
    return raw_yaml

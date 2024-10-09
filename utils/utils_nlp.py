import re

def split_paragraph(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    sentences = [sentence for sentence in sentences if sentence]
    sentence_count = len(sentences)
    return sentences, sentence_count
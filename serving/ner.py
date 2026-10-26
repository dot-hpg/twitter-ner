"""Inference for the fine-tuned BERT Twitter NER model (same preprocessing and sub-token joining as notebook 02)."""
import os
import re

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from transformers import AutoTokenizer, TFAutoModelForTokenClassification  # noqa: E402

MAX_LEN = 128


def bert_normalise(word):
    """URLs and @mentions are replaced by short placeholders, exactly as during training."""
    if re.match(r"https?://|www\.", word, re.I):
        return "url"
    if word.startswith("@") and len(word) > 1:
        return "@user"
    return word


def repair_bio(tags):
    """Turn an I- tag that does not continue an entity of the same type into a B- tag."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        out.append(t)
        prev = t
    return out


def to_entities(words, tags):
    """Group BIO tags into entity spans with word offsets."""
    ents, start, typ = [], None, None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or t[2:] != typ:
            if start is not None:
                ents.append({"text": " ".join(words[start:i]), "type": typ, "start": start, "end": i})
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return ents


class NerModel:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = TFAutoModelForTokenClassification.from_pretrained(model_dir)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, texts):
        """Tag a batch of raw texts; returns words, word-level BIO tags and entity spans for each."""
        batch_words = [t.split() for t in texts]
        enc = self.tokenizer([[bert_normalise(w) for w in ws] for ws in batch_words], is_split_into_words=True,
                             truncation=True, max_length=MAX_LEN, padding=True, return_tensors="tf")
        logits = self.model(dict(enc), training=False).logits.numpy()
        results = []
        for b, words in enumerate(batch_words):
            ids, tags, prev = np.argmax(logits[b], axis=-1), [], None
            for pos, w in enumerate(enc.word_ids(b)):
                if w is not None and w != prev:          # first sub-token of each word carries the word's label
                    tags.append(self.id2label[int(ids[pos])])
                prev = w
            tags = repair_bio(tags + ["O"] * (len(words) - len(tags)))  # words cut by truncation get "O"
            results.append({"words": words, "tags": tags, "entities": to_entities(words, tags)})
        return results

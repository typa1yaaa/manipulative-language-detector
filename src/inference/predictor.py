from pathlib import Path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import os 

from ..training import config


SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


class ManipulationDetector:
    def __init__(self, model_path: str | Path, threshold: float = config.PREDICT_THRESHOLD):
        model_path = model_path or config.get_model_source()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.threshold = threshold
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}


    @torch.no_grad()
    def _token_predictions(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LENGTH
        )
        logits = self.model(**inputs).logits[0]
        probs = torch.sigmoid(logits)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return tokens, probs
    

    def _words_from_tokens(self, tokens, probs):
        token_labels, token_probs = [], []
        for i, tok in enumerate(tokens):
            if tok in SPECIAL_TOKENS:
                token_labels.append([])
                token_probs.append([])
                continue
            active = (probs[i] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
            token_labels.append([self.id2label[j] for j in active])
            token_probs.append([probs[i, j].item() for j in active])
 
        words, word_labels, word_probs = [], [], []
        cur_word, cur_labels, cur_probs = "", [], []
        for tok, labs, p in zip(tokens, token_labels, token_probs):
            if tok in SPECIAL_TOKENS:
                continue
            if tok.startswith("##"):
                cur_word += tok[2:]
            else:
                if cur_word:
                    words.append(cur_word)
                    word_labels.append(cur_labels)
                    word_probs.append(cur_probs)
                cur_word, cur_labels, cur_probs = tok, labs, p
        if cur_word:
            words.append(cur_word)
            word_labels.append(cur_labels)
            word_probs.append(cur_probs)
        return words, word_labels, word_probs


    def predict(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []
 
        tokens, probs = self._token_predictions(text)
        words, word_labels, word_probs = self._words_from_tokens(tokens, probs)
 
        spans = []
        for label_name in self.id2label.values():
            i = 0
            while i < len(words):
                if label_name in word_labels[i]:
                    span_words = [words[i]]
                    max_prob = word_probs[i][word_labels[i].index(label_name)]
                    i += 1
                    while i < len(words) and label_name in word_labels[i]:
                        span_words.append(words[i])
                        max_prob = max(max_prob, word_probs[i][word_labels[i].index(label_name)])
                        i += 1
                    spans.append({
                        "text": " ".join(span_words),
                        "pattern_name": label_name,
                        "confidence": max_prob,
                    })
                else:
                    i += 1
        return spans


    def explain(self, text: str) -> str:
        spans = self.predict(text)
        lines = [f"\nТЕКСТ:\n{text}\n\nРЕЗУЛЬТАТ АНАЛИЗА:"]
        if not spans:
            lines.append("МАНИПУЛЯЦИЯ НЕ ОБНАРУЖЕНА\n")
            return "\n".join(lines)
 
        lines.append("МАНИПУЛЯЦИЯ ОБНАРУЖЕНА\n")
        patterns: dict[str, list[dict]] = {}
        for s in spans:
            patterns.setdefault(s["pattern_name"], []).append(s)
 
        for idx, (pname, plist) in enumerate(patterns.items(), 1):
            ru_name = config.LABEL_RU.get(pname, pname)
            lines.append(f"{idx}. Манипуляция {pname} ({ru_name})")
            lines.append("      Примеры:")
            for s in plist:
                lines.append(f'        - "{s["text"]}" (conf: {s["confidence"]*100:.2f}%)')
            lines.append("")
        return "\n".join(lines)
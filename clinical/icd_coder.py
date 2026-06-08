"""Automated ICD-10 coding from clinical notes."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import List, Tuple

class ClinicalICD10Coder:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        with open(f"{model_path}/icd10_labels.txt") as f:
            self.labels = [l.strip() for l in f]

    def deidentify(self, text: str) -> str:
        """Remove PHI before processing."""
        text = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]", text)
        text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]", text)
        text = re.sub(r"\b\d{2}/\d{2}/\d{4}\b", "[DATE]", text)
        return text

    def predict_codes(self, clinical_note: str,
                      top_k: int = 5, threshold: float = 0.3) -> List[Tuple[str, float]]:
        clean_note = self.deidentify(clinical_note)
        inputs = self.tokenizer(clean_note, return_tensors="pt",
                                max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits)[0].numpy()
        top_indices = probs.argsort()[-top_k:][::-1]
        return [(self.labels[i], float(probs[i]))
                for i in top_indices if probs[i] >= threshold]

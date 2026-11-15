"""Predict a binary label for an already-cleaned overlapping caller utterance."""

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--model", default="HashNuke/barge-in-detector")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).cpu().eval()
    with torch.inference_mode():
        enc = tokenizer(args.text, truncation=True, max_length=64, return_tensors="pt")
        p = torch.softmax(model(**enc).logits, dim=-1)[0, model.config.label2id["barge_in"]].item()
    print(json.dumps({"p_barge_in": p, "label": "barge_in" if p >= 0.5 else "backchannel"}))


if __name__ == "__main__":
    main()

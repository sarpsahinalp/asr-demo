import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)


DATA_PATH = "intent/dataset.jsonl"
INTENT_MODEL_DIR = "intent/hf_intent_model"
SLOT_MODEL_DIR = "intent/hf_slot_model"
BASE_MODEL = "distilbert-base-uncased"


@dataclass
class Example:
    text: str
    intent: str
    tokens: List[str]
    slots: List[str]


def load_dataset(path: str) -> List[Example]:
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            intent = obj["intent"]
            slots = obj["slots"]
            tokens = [s["token"] for s in slots]
            slot_tags = [s["tag"] for s in slots]
            examples.append(Example(text=text, intent=intent, tokens=tokens, slots=slot_tags))
    return examples


def build_label_maps(examples: List[Example]):
    intent_labels = sorted({ex.intent for ex in examples})
    slot_labels = sorted({tag for ex in examples for tag in ex.slots})

    intent2id = {label: i for i, label in enumerate(intent_labels)}
    id2intent = {i: label for label, i in intent2id.items()}

    slot2id = {label: i for i, label in enumerate(slot_labels)}
    id2slot = {i: label for label, i in slot2id.items()}

    return intent2id, id2intent, slot2id, id2slot


def prepare_intent_dataset(examples: List[Example], tokenizer, intent2id: Dict[str, int]) -> Dataset:
    texts = [ex.text for ex in examples]
    intents = [intent2id[ex.intent] for ex in examples]

    encodings = tokenizer(texts, padding=True, truncation=True)
    encodings["labels"] = intents
    return Dataset.from_dict(encodings)


def align_slots_with_tokens(
    tokenizer, examples: List[Example], slot2id: Dict[str, int]
) -> Dataset:
    tokenized_inputs = tokenizer(
        [ex.tokens for ex in examples],
        is_split_into_words=True,
        padding=True,
        truncation=True,
    )

    all_labels: List[List[int]] = []
    for i, ex in enumerate(examples):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids: List[int] = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                # Assign the label of the first sub-token, ignore the rest
                if word_idx != previous_word_idx:
                    label_ids.append(slot2id[ex.slots[word_idx]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return Dataset.from_dict(tokenized_inputs)


def train_intent_model(train_dataset: Dataset, num_labels: int, id2label: Dict[int, str]):
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )

    args = TrainingArguments(
        output_dir="intent/hf_intent_output",
        per_device_train_batch_size=8,
        num_train_epochs=10,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(INTENT_MODEL_DIR)


def train_slot_model(train_dataset: Dataset, num_labels: int, id2label: Dict[int, str]):
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )

    args = TrainingArguments(
        output_dir="intent/hf_slot_output",
        per_device_train_batch_size=8,
        num_train_epochs=10,
        learning_rate=5e-5,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(SLOT_MODEL_DIR)


def main():
    print("Loading dataset...")
    examples = load_dataset(DATA_PATH)

    print(f"Loaded {len(examples)} examples.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Building label maps...")
    intent2id, id2intent, slot2id, id2slot = build_label_maps(examples)

    print("Preparing intent dataset...")
    intent_dataset = prepare_intent_dataset(examples, tokenizer, intent2id)

    print("Preparing slot dataset...")
    slot_dataset = align_slots_with_tokens(tokenizer, examples, slot2id)

    print("Training intent model...")
    train_intent_model(intent_dataset, num_labels=len(intent2id), id2label=id2intent)

    print("Training slot model...")
    train_slot_model(slot_dataset, num_labels=len(slot2id), id2label=id2slot)

    print(f"Saving tokenizer to {INTENT_MODEL_DIR} and {SLOT_MODEL_DIR}...")
    tokenizer.save_pretrained(INTENT_MODEL_DIR)
    tokenizer.save_pretrained(SLOT_MODEL_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
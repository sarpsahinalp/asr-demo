import speech_recognition as sr
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
import torch

INTENT_MODEL_DIR = "intent/hf_intent_model"
SLOT_MODEL_DIR = "intent/hf_slot_model"

print("Loading NLU models...")
tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_DIR)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR)
slot_model = AutoModelForTokenClassification.from_pretrained(SLOT_MODEL_DIR)
intent_model.eval()
slot_model.eval()
print("NLU models loaded.")

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    print("\nCalibrating for ambient noise... Please wait a moment.")
    r.adjust_for_ambient_noise(source, duration=2)
    print("Calibration complete. You can start speaking.")

while True:
    try:
        print("\nListening...")
        with mic as source:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)

        print("Processing audio...")

        start_time = time.time()
        transcript = r.recognize_whisper(audio, language="english")
        asr_time = time.time() - start_time

        print(f"Transcript: '{transcript}' (in {asr_time:.2f}s)")

        # NLU: Intent + Slots
        start_time = time.time()

        with torch.no_grad():
            # Intent prediction
            encoded_intent = tokenizer(
                transcript,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            intent_outputs = intent_model(**encoded_intent)
            intent_logits = intent_outputs.logits
            intent_id = int(intent_logits.argmax(dim=-1).item())
            intent_label = intent_model.config.id2label[intent_id]

            # Slot prediction
            encoded_slots = tokenizer(
                transcript.split(),
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            slot_outputs = slot_model(**encoded_slots)
            slot_logits = slot_outputs.logits
            slot_ids = slot_logits.argmax(dim=-1)[0].tolist()

            # Align predicted slot labels with input tokens
            word_ids = encoded_slots.word_ids(0)
            slots = []
            last_word_idx = None
            for token_id, word_idx in zip(slot_ids, word_ids):
                if word_idx is None:
                    continue
                if word_idx == last_word_idx:
                    continue
                token = transcript.split()[word_idx]
                label = slot_model.config.id2label[int(token_id)]
                slots.append((token, label))
                last_word_idx = word_idx

        nlu_time = time.time() - start_time

        print(f"--> Intent: {intent_label} (in {nlu_time * 1000:.2f}ms)")
        print("--> Slots:")
        for token, label in slots:
            print(f"   {token}: {label}")

    except sr.WaitTimeoutError:
        print("No speech detected. Listening again...")
    except sr.UnknownValueError:
        print("Whisper could not understand audio. Please try again.")
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break
import speech_recognition as sr
import joblib
import time

print("Loading NLU model...")
nlu_model = joblib.load("intent/nlu_intent_model.joblib")
print("NLU model loaded.")

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

        start_time = time.time()
        intent = nlu_model.predict([transcript])[0]
        nlu_time = time.time() - start_time

        print(f"--> Intent: {intent} (in {nlu_time * 1000:.2f}ms)")

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
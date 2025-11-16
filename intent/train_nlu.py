import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

texts = [
    "what's the weather like today",
    "is it going to rain tomorrow",
    "turn on the lights in the living room",
    "switch on the kitchen light",
    "play some music",
    "who is the CEO of Google",
    "what is a transformer model"
]

intents = [
    "GetWeather",
    "GetWeather",
    "ControlLights",
    "ControlLights",
    "PlayMusic",
    "GetFact",
    "GetFact"
]

nlu_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='lbfgs', multi_class='auto'))
])

print("Training NLU model...")
nlu_pipeline.fit(texts, intents)


model_filename = "nlu_intent_model.joblib"
joblib.dump(nlu_pipeline, model_filename)

print(f"Model saved to {model_filename}!")
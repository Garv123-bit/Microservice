from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import numpy as np
import librosa
import tempfile
import os

# Whisper (free local model)
import whisper

app = Flask(__name__)
encoder = VoiceEncoder()
model = whisper.load_model("small")  # free local Whisper model

# --- Voice embedding endpoint ---
@app.route('/embed', methods=['POST'])
def embed_voice():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            wav, _ = librosa.load(tmp.name, sr=16000)
            embedding = encoder.embed_utterance(wav)
        os.remove(tmp.name)
        return jsonify({'embedding': embedding.tolist()})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

# --- Voice verification endpoint ---
@app.route('/verify', methods=['POST'])
def verify_voice():
    try:
        data = request.get_json()
        emb1 = np.array(data['embedding1'])
        emb2 = np.array(data['embedding2'])
        similarity = 1 - cosine(emb1, emb2)
        is_match = similarity >= 0.75
        return jsonify({'similarity': float(similarity), 'match': is_match})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

# --- Whisper transcription endpoint ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            result = model.transcribe(tmp.name)
        os.remove(tmp.name)
        return jsonify({'transcript': result['text']})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

# --- Home route ---
@app.route('/')
def home():
    return "Voice Microservice + Whisper Active ✅"

# --- Run server ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)

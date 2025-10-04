from flask import Flask, request, jsonify
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import numpy as np
import librosa
import tempfile
import os

app = Flask(__name__)
encoder = VoiceEncoder()

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

@app.route('/verify', methods=['POST'])
def verify_voice():
    try:
        data = request.get_json()
        emb1 = np.array(data['embedding1'])
        emb2 = np.array(data['embedding2'])
        similarity = 1 - cosine(emb1, emb2)
        is_match = similarity >= 0.75  # threshold
        return jsonify({'similarity': float(similarity), 'match': is_match})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Voice Microservice Active ✅"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)

from flask import Flask, request, jsonify
from flask_cors import CORS
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine
import numpy as np
import librosa
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Lazy init (prevents crash at cold start)
encoder = None

def get_encoder():
    global encoder
    if encoder is None:
        encoder = VoiceEncoder()
    return encoder

@app.route('/embed', methods=['POST'])
def embed_voice():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            # Load smaller chunks → reduce RAM spike
            wav, _ = librosa.load(tmp.name, sr=16000, mono=True)
            wav = wav[:16000 * 20]  # limit to first 20 seconds (safe)
            emb = get_encoder().embed_utterance(wav)
        os.remove(tmp.name)
        return jsonify({'embedding': emb.tolist()})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_voice():
    try:
        data = request.get_json()
        emb1 = np.array(data['embedding1'])
        emb2 = np.array(data['embedding2'])
        sim = 1 - cosine(emb1, emb2)
        return jsonify({'similarity': float(sim), 'match': sim >= 0.75})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Voice Microservice Active ✅"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

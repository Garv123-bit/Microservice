from flask import Flask, request, jsonify
from flask_cors import CORS
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine
import numpy as np
import librosa
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
encoder = VoiceEncoder()

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
        is_match = similarity >= 0.75
        return jsonify({'similarity': float(similarity), 'match': is_match})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Voice Microservice Active ✅"

# Local dev only
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

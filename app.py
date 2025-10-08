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
encoder = VoiceEncoder()

# Max chunk duration in seconds
MAX_CHUNK_SEC = 5  

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
            tmp_path = tmp.name

        # Load full audio
        wav, sr = librosa.load(tmp_path, sr=16000)
        os.remove(tmp_path)

        # Break into chunks
        embeddings = []
        total_samples = wav.shape[0]
        chunk_samples = int(MAX_CHUNK_SEC * sr)
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk_wav = wav[start:end]
            chunk_emb = encoder.embed_utterance(chunk_wav)
            embeddings.append(chunk_emb)

        # Average all chunk embeddings
        final_embedding = np.mean(np.stack(embeddings), axis=0)

        return jsonify({'embedding': final_embedding.tolist()})

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

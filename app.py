from flask import Flask, request, jsonify
from flask_cors import CORS
from resemblyzer import VoiceEncoder
from scipy.spatial.distance import cosine
import numpy as np
import librosa
import tempfile
import os
import logging

app = Flask(__name__)
CORS(app)

# Logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-ms")

encoder = VoiceEncoder()

# Max chunk duration in seconds
MAX_CHUNK_SEC = 5  

@app.route('/embed', methods=['POST'])
def embed_voice():
    try:
        logger.info("POST /embed hit")
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Load full audio
        wav, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.remove(tmp_path)

        # Chunking
        chunk_len = sr * MAX_CHUNK_SEC
        embeddings = []

        for start in range(0, len(wav), chunk_len):
            chunk = wav[start:start + chunk_len]
            if len(chunk) == 0:
                continue
            emb = encoder.embed_utterance(chunk)
            embeddings.append(emb)

        if not embeddings:
            return jsonify({'error': 'Audio too short or empty'}), 400

        final_embedding = np.mean(np.vstack(embeddings), axis=0)
        return jsonify({'embedding': final_embedding.tolist()})

    except Exception as e:
        logger.error("Error in /embed: %s", str(e))
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
        logger.error("Error in /verify: %s", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return "Voice Microservice Active ✅"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask server on port %d", port)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

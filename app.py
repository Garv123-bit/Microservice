import os
import tempfile
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# optional - add psutil to requirements for memory logging
try:
    import psutil
except Exception:
    psutil = None

import librosa
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-ms")

# encoder is lazy-initialized to avoid loading heavy model at import/start
encoder = None
def get_encoder():
    global encoder
    if encoder is None:
        logger.info("Loading VoiceEncoder now (lazy)...")
        # Import here to slightly delay import-time cost
        from resemblyzer import VoiceEncoder
        encoder = VoiceEncoder()
        logger.info("VoiceEncoder loaded.")
    return encoder

def log_mem(prefix=""):
    if psutil:
        p = psutil.Process()
        mem = p.memory_info().rss / (1024*1024)
        logger.info(f"{prefix} RSS memory: {mem:.1f} MB")
    else:
        logger.info(f"{prefix} (psutil not installed — cannot log memory)")

@app.route('/')
def home():
    return "Voice Microservice Active ✅"

@app.route('/embed', methods=['POST'])
def embed_voice():
    logger.info("POST /embed hit")
    log_mem("before")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)
        # limit load duration to reduce memory pressure (in seconds)
        max_seconds = 20  # adjust if needed; keep modest on small RAM
        wav, _ = librosa.load(tmp_path, sr=16000, mono=True, duration=max_seconds)
        enc = get_encoder()
        # embedding (this is the heavy part)
        embedding = enc.embed_utterance(wav)
        log_mem("after")
        return jsonify({'embedding': embedding.tolist()})
    except Exception as e:
        logger.exception("Embed error")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

@app.route('/verify', methods=['POST'])
def verify_voice():
    try:
        data = request.get_json()
        emb1 = np.array(data['embedding1'])
        emb2 = np.array(data['embedding2'])
        similarity = 1 - cosine(emb1, emb2)
        return jsonify({'similarity': float(similarity), 'match': similarity >= 0.75})
    except Exception as e:
        logger.exception("Verify error")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

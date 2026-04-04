from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import traceback
import time

# Reduce TensorFlow thread usage on small hosted servers
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

# Use threading mode for hosted CPU-heavy inference
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

print("Loading model...")
model = load_model("music_model_emotion.h5")
print("Model loaded successfully.")

# Warm up model once at startup
try:
    warmup_input = np.array([[60, 62, 64, 65, 67, 69, 71, 72, 0, 1, 0, 0]], dtype=np.float32)
    warmup_start = time.time()
    warmup_output = model(warmup_input, training=False).numpy()
    warmup_end = time.time()
    print(f"Warmup complete. Output shape={warmup_output.shape}. Took {warmup_end - warmup_start:.3f}s")
except Exception as e:
    print(f"Warmup failed: {e}")
    traceback.print_exc()

emotion_map = {
    "relax": [1, 0, 0, 0],
    "happy": [0, 1, 0, 0],
    "sad":   [0, 0, 1, 0],
    "focus": [0, 0, 0, 1]
}

def detect_emotion(user_text):
    text = user_text.lower()
    if "stress" in text or "tired" in text:
        return emotion_map["relax"], "relax"
    elif "happy" in text or "excited" in text:
        return emotion_map["happy"], "happy"
    elif "sad" in text or "down" in text:
        return emotion_map["sad"], "sad"
    else:
        return emotion_map["focus"], "focus"

def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions = np.log(predictions + 1e-9) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)

client_sequences = {}
client_emotions = {}

def run_inference(x_input):
    start = time.time()
    output = model(x_input, training=False).numpy()[0]
    end = time.time()
    print(f"[run_inference] finished in {end - start:.3f}s")
    return output

def generate_chunk(seed_seq, emotion_vector, chunk_size=4):
    print(f"generate_chunk called with seed_seq={seed_seq}, emotion_vector={emotion_vector}, chunk_size={chunk_size}")

    pattern = list(seed_seq)
    generated_notes = []

    for i in range(chunk_size):
        x_input = np.array([pattern + emotion_vector], dtype=np.float32)

        print(f"[generate_chunk] iteration={i}")
        print(f"[generate_chunk] x_input.shape={x_input.shape}")
        print(f"[generate_chunk] x_input={x_input}")

        prediction = run_inference(x_input)
        print(f"[generate_chunk] raw prediction shape={prediction.shape}")

        prediction_sum = prediction.sum()
        if prediction_sum <= 0:
            raise ValueError("Prediction sum is zero or negative.")

        prediction = prediction / prediction_sum

        note_idx = sample(prediction)
        note_idx = int(max(21, min(108, note_idx)))

        generated_notes.append(note_idx)

        pattern.append(note_idx)
        pattern = pattern[1:]

    print(f"Generated notes: {generated_notes}")
    return pattern, generated_notes

@app.route("/")
def home():
    return "Flask SocketIO server is running."

@socketio.on("connect")
def handle_connect():
    print(f"Client connected: sid={request.sid}")

@socketio.on("start_music")
def handle_start(data):
    sid = request.sid
    print(f"[start_music] received from sid={sid}, data={data}")

    try:
        user_text = data.get("user_text", "")
        if not user_text:
            emit("error", {"message": "user_text is required"})
            return

        emotion_vector, emotion_label = detect_emotion(user_text)
        print(f"[start_music] detected emotion={emotion_label}, vector={emotion_vector}")

        client_sequences[sid] = [60, 62, 64, 65, 67, 69, 71, 72]
        client_emotions[sid] = (emotion_vector, emotion_label)

        seed_sequence = client_sequences[sid]
        seed_sequence, notes_chunk = generate_chunk(seed_sequence, emotion_vector, chunk_size=8)
        client_sequences[sid] = seed_sequence

        print(f"[start_music] emitting new_notes={notes_chunk}")
        emit("new_notes", {"notes": notes_chunk, "emotion": emotion_label})

    except Exception as e:
        print(f"[start_music] ERROR: {e}")
        traceback.print_exc()
        emit("error", {"message": f"Server exception: {str(e)}"})

@socketio.on("request_more")
def handle_request_more(data):
    sid = request.sid
    print(f"[request_more] received from sid={sid}, data={data}")

    try:
        if sid not in client_sequences:
            emit("error", {"message": "Session not initialized"})
            return

        user_text = data.get("user_text", "")
        if user_text:
            emotion_vector, emotion_label = detect_emotion(user_text)
            client_emotions[sid] = (emotion_vector, emotion_label)
        else:
            emotion_vector, emotion_label = client_emotions[sid]

        seed_sequence = client_sequences[sid]
        seed_sequence, notes_chunk = generate_chunk(seed_sequence, emotion_vector, chunk_size=8)
        client_sequences[sid] = seed_sequence

        print(f"[request_more] emitting new_notes={notes_chunk}")
        emit("new_notes", {"notes": notes_chunk, "emotion": emotion_label})

    except Exception as e:
        print(f"[request_more] ERROR: {e}")
        traceback.print_exc()
        emit("error", {"message": f"Server exception: {str(e)}"})

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: sid={sid}")
    client_sequences.pop(sid, None)
    client_emotions.pop(sid, None)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5003)))
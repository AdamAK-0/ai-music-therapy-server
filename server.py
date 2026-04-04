from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = load_model("music_model_emotion.h5")

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
    predictions = np.log(predictions + 1e-9) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(predictions), p=predictions)

client_sequences = {}  # sid: seed sequence
client_emotions = {}   # sid: (vector, label)

def generate_chunk(seed_seq, emotion_vector, chunk_size=8):
    pattern = list(seed_seq)
    generated_notes = []

    for _ in range(chunk_size):

        x_input = np.array(pattern + emotion_vector).reshape(
            1, len(pattern) + len(emotion_vector)
        )

        prediction = model.predict(x_input, verbose=0)[0]

        prediction = prediction / prediction.sum()

        note_idx = sample(prediction)

        note_idx = int(max(21, min(108, note_idx)))

        generated_notes.append(note_idx)

        pattern.append(note_idx)
        pattern = pattern[1:]

    return pattern, generated_notes

@socketio.on("start_music")
def handle_start(data):
    sid = request.sid
    user_text = data.get("user_text", "")
    if not user_text:
        emit("error", {"message": "user_text is required"})
        return

    emotion_vector, emotion_label = detect_emotion(user_text)
    client_sequences[sid] = [60, 62, 64, 65, 67, 69, 71, 72]  # initial seed
    client_emotions[sid] = (emotion_vector, emotion_label)

    seed_sequence = client_sequences[sid]
    seed_sequence, notes_chunk = generate_chunk(seed_sequence, emotion_vector)
    client_sequences[sid] = seed_sequence
    emit("new_notes", {"notes": notes_chunk, "emotion": emotion_label})

@socketio.on("request_more")
def handle_request_more(data):
    sid = request.sid
    if sid not in client_sequences:
        emit("error", {"message": "Session not initialized"})
        return

    # Use the latest user text to update emotion dynamically
    user_text = data.get("user_text", "")
    if user_text:
        emotion_vector, emotion_label = detect_emotion(user_text)
        client_emotions[sid] = (emotion_vector, emotion_label)
    else:
        emotion_vector, emotion_label = client_emotions[sid]

    seed_sequence = client_sequences[sid]
    seed_sequence, notes_chunk = generate_chunk(seed_sequence, emotion_vector)
    client_sequences[sid] = seed_sequence
    emit("new_notes", {"notes": notes_chunk, "emotion": emotion_label})

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    if sid in client_sequences:
        del client_sequences[sid]
    if sid in client_emotions:
        del client_emotions[sid]

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5003)
from flask import Flask, request
from flask_socketio import SocketIO, emit
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import traceback
import time

# Reduce TensorFlow thread usage on small hosted servers.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

MIN_MIDI = 21
MAX_MIDI = 108
MODEL_SEED_SIZE = 8
DEFAULT_CHUNK_SIZE = int(os.environ.get("MUSIC_CHUNK_SIZE", "8"))

EMOTION_MAP = {
    "relax": [1, 0, 0, 0],
    "happy": [0, 1, 0, 0],
    "sad": [0, 0, 1, 0],
    "focus": [0, 0, 0, 1],
}

EMOTION_CONFIG = {
    "relax": {
        "seed": [60, 64, 67, 71, 69, 67, 64, 62],
        "range": (48, 76),
        "scale": [0, 2, 4, 7, 9],
        "temperature": 0.82,
        "max_leap": 7,
    },
    "happy": {
        "seed": [60, 62, 64, 67, 69, 72, 74, 76],
        "range": (55, 84),
        "scale": [0, 2, 4, 5, 7, 9, 11],
        "temperature": 0.95,
        "max_leap": 9,
    },
    "sad": {
        "seed": [57, 60, 62, 64, 65, 64, 62, 60],
        "range": (45, 74),
        "scale": [0, 2, 3, 5, 7, 8, 10],
        "temperature": 0.78,
        "max_leap": 6,
    },
    "focus": {
        "seed": [60, 67, 64, 67, 62, 69, 65, 69],
        "range": (50, 79),
        "scale": [0, 2, 4, 7, 9],
        "temperature": 0.72,
        "max_leap": 5,
    },
}

KEY_ROOTS = {
    "relax": 60,
    "happy": 60,
    "sad": 57,
    "focus": 60,
}

TEXT_EMOTION_HINTS = {
    "relax": ["stress", "stressed", "anxious", "tired", "calm", "relax", "sleep"],
    "happy": ["happy", "excited", "joy", "great", "celebrate", "upbeat"],
    "sad": ["sad", "down", "lonely", "cry", "grief", "heartbroken"],
    "focus": ["focus", "study", "work", "concentrate", "productive", "coding"],
}

print("Loading model...")
model = load_model("music_model_emotion.h5")
print("Model loaded successfully.")


def warm_up_model():
    try:
        warmup_input = np.array(
            [[60, 62, 64, 65, 67, 69, 71, 72, 0, 1, 0, 0]],
            dtype=np.float32,
        )
        warmup_start = time.time()
        warmup_output = model(warmup_input, training=False).numpy()
        warmup_end = time.time()
        print(
            "Warmup complete. "
            f"Output shape={warmup_output.shape}. Took {warmup_end - warmup_start:.3f}s"
        )
    except Exception as exc:
        print(f"Warmup failed: {exc}")
        traceback.print_exc()


warm_up_model()

client_sequences = {}
client_emotions = {}


def normalize_emotion_label(value):
    label = str(value or "").strip().lower()
    return label if label in EMOTION_MAP else None


def detect_emotion_from_text(user_text):
    text = str(user_text or "").lower()
    scores = {emotion: 0 for emotion in EMOTION_MAP}

    for emotion, hints in TEXT_EMOTION_HINTS.items():
        for hint in hints:
            if hint in text:
                scores[emotion] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "focus"

    return EMOTION_MAP[best], best


def resolve_emotion(data):
    explicit_label = normalize_emotion_label(data.get("emotion"))
    if explicit_label:
        return EMOTION_MAP[explicit_label], explicit_label

    return detect_emotion_from_text(data.get("user_text", ""))


def fold_to_range(note, low, high):
    note = int(round(note))

    while note < low:
        note += 12
    while note > high:
        note -= 12

    return max(low, min(high, note))


def quantize_to_scale(note, emotion_label):
    config = EMOTION_CONFIG[emotion_label]
    low, high = config["range"]
    scale = config["scale"]
    root = KEY_ROOTS[emotion_label]

    candidates = []
    for octave in range(-3, 7):
        for degree in scale:
            candidate = root + (12 * octave) + degree
            if low <= candidate <= high:
                candidates.append(candidate)

    if not candidates:
        return fold_to_range(note, low, high)

    return min(candidates, key=lambda candidate: abs(candidate - note))


def avoid_repetition(note, previous_note, emotion_label, repeat_count):
    if previous_note is None:
        return note

    config = EMOTION_CONFIG[emotion_label]
    low, high = config["range"]

    if note == previous_note and repeat_count >= 1:
        direction = 1 if previous_note < (low + high) / 2 else -1
        note = quantize_to_scale(previous_note + direction * 2, emotion_label)

    if abs(note - previous_note) > config["max_leap"]:
        while note - previous_note > config["max_leap"]:
            note -= 12
        while previous_note - note > config["max_leap"]:
            note += 12
        note = quantize_to_scale(fold_to_range(note, low, high), emotion_label)

    return max(low, min(high, note))


def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions, dtype=np.float64)
    predictions = np.maximum(predictions, 1e-9)
    predictions = np.log(predictions) / max(temperature, 0.05)
    exp_preds = np.exp(predictions - np.max(predictions))
    probabilities = exp_preds / np.sum(exp_preds)
    return int(np.random.choice(len(probabilities), p=probabilities))


def run_inference(x_input):
    start = time.time()
    output = model(x_input, training=False).numpy()[0]
    end = time.time()
    print(f"[run_inference] finished in {end - start:.3f}s")
    return output


def shape_model_note(raw_note, previous_note, emotion_label, repeat_count):
    config = EMOTION_CONFIG[emotion_label]
    low, high = config["range"]
    note = fold_to_range(raw_note, low, high)
    note = quantize_to_scale(note, emotion_label)
    note = avoid_repetition(note, previous_note, emotion_label, repeat_count)
    return int(max(MIN_MIDI, min(MAX_MIDI, note)))


def generate_chunk(seed_seq, emotion_vector, emotion_label, chunk_size=DEFAULT_CHUNK_SIZE):
    print(
        "[generate_chunk] "
        f"emotion={emotion_label}, seed_seq={seed_seq}, chunk_size={chunk_size}"
    )

    pattern = list(seed_seq)[-MODEL_SEED_SIZE:]
    generated_notes = []
    previous_note = pattern[-1] if pattern else None
    repeat_count = 0
    temperature = EMOTION_CONFIG[emotion_label]["temperature"]

    for i in range(chunk_size):
        x_input = np.array([pattern + emotion_vector], dtype=np.float32)
        prediction = run_inference(x_input)

        prediction_sum = float(prediction.sum())
        if prediction_sum <= 0:
            raise ValueError("Prediction sum is zero or negative.")

        prediction = prediction / prediction_sum
        raw_note = sample(prediction, temperature=temperature)
        note = shape_model_note(raw_note, previous_note, emotion_label, repeat_count)

        if note == previous_note:
            repeat_count += 1
        else:
            repeat_count = 0

        generated_notes.append(note)
        pattern.append(note)
        pattern = pattern[-MODEL_SEED_SIZE:]
        previous_note = note

        print(f"[generate_chunk] i={i}, raw_note={raw_note}, shaped_note={note}")

    print(f"Generated notes: {generated_notes}")
    return pattern, generated_notes


@app.route("/")
def home():
    return "AI Music Therapy SocketIO server is running."


@app.route("/health")
def health():
    return {"status": "ok", "chunk_size": DEFAULT_CHUNK_SIZE}


@socketio.on("connect")
def handle_connect():
    print(f"Client connected: sid={request.sid}")


@socketio.on("start_music")
def handle_start(data):
    sid = request.sid
    data = data or {}
    print(f"[start_music] received from sid={sid}, data={data}")

    try:
        user_text = data.get("user_text", "")
        if not user_text:
            emit("error", {"message": "user_text is required"})
            return

        emotion_vector, emotion_label = resolve_emotion(data)
        print(f"[start_music] emotion={emotion_label}, vector={emotion_vector}")

        client_sequences[sid] = EMOTION_CONFIG[emotion_label]["seed"].copy()
        client_emotions[sid] = (emotion_vector, emotion_label)

        seed_sequence, notes_chunk = generate_chunk(
            client_sequences[sid],
            emotion_vector,
            emotion_label,
        )
        client_sequences[sid] = seed_sequence

        emit("new_notes", {"notes": notes_chunk, "emotion": emotion_label})

    except Exception as exc:
        print(f"[start_music] ERROR: {exc}")
        traceback.print_exc()
        emit("error", {"message": f"Server exception: {str(exc)}"})


@socketio.on("request_more")
def handle_request_more(data):
    sid = request.sid
    data = data or {}
    print(f"[request_more] received from sid={sid}, data={data}")

    try:
        if sid not in client_sequences:
            emit("error", {"message": "Session not initialized"})
            return

        emotion_vector, emotion_label = resolve_emotion(data)
        client_emotions[sid] = (emotion_vector, emotion_label)

        seed_sequence, notes_chunk = generate_chunk(
            client_sequences[sid],
            emotion_vector,
            emotion_label,
        )
        client_sequences[sid] = seed_sequence

        emit("new_notes", {"notes": notes_chunk, "emotion": emotion_label})

    except Exception as exc:
        print(f"[request_more] ERROR: {exc}")
        traceback.print_exc()
        emit("error", {"message": f"Server exception: {str(exc)}"})


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: sid={sid}")
    client_sequences.pop(sid, None)
    client_emotions.pop(sid, None)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5003)))

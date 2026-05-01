import argparse
import json
import math
import os
import time
import urllib.request
import zipfile
from itertools import combinations
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.models import load_model


MAESTRO_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)

MODEL_PATH = "music_model_emotion.h5"
ARTIFACT_DIR = Path("evaluation_artifacts")
MAESTRO_DIR = Path("maestro-v3.0.0")
MAESTRO_ZIP = Path("maestro-v3.0.0-midi.zip")

MIN_MIDI = 21
MAX_MIDI = 108
TRAIN_NOTE_CUTOFF = 1_000_000

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
        "root": 60,
    },
    "happy": {
        "seed": [60, 62, 64, 67, 69, 72, 74, 76],
        "range": (55, 84),
        "scale": [0, 2, 4, 5, 7, 9, 11],
        "temperature": 0.95,
        "max_leap": 9,
        "root": 60,
    },
    "sad": {
        "seed": [57, 60, 62, 64, 65, 64, 62, 60],
        "range": (45, 74),
        "scale": [0, 2, 3, 5, 7, 8, 10],
        "temperature": 0.78,
        "max_leap": 6,
        "root": 57,
    },
    "focus": {
        "seed": [60, 67, 64, 67, 62, 69, 65, 69],
        "range": (50, 79),
        "scale": [0, 2, 4, 7, 9],
        "temperature": 0.72,
        "max_leap": 5,
        "root": 60,
    },
}


def ensure_maestro():
    if MAESTRO_DIR.exists():
        return

    if not MAESTRO_ZIP.exists():
        print(f"Downloading {MAESTRO_URL}")
        urllib.request.urlretrieve(MAESTRO_URL, MAESTRO_ZIP)

    print(f"Extracting {MAESTRO_ZIP}")
    with zipfile.ZipFile(MAESTRO_ZIP, "r") as archive:
        archive.extractall(".")


def extract_notes(max_notes):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    cache_path = ARTIFACT_DIR / f"maestro_notes_first_{max_notes}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    ensure_maestro()
    midi_files = sorted(
        [
            path
            for path in MAESTRO_DIR.rglob("*")
            if path.suffix.lower() in {".mid", ".midi"}
        ]
    )

    notes = []
    for index, midi_path in enumerate(midi_files, start=1):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:
            print(f"Skipping unreadable MIDI {midi_path}: {exc}")
            continue

        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append(int(note.pitch))
                if len(notes) >= max_notes:
                    break
            if len(notes) >= max_notes:
                break

        if index % 100 == 0:
            print(f"Parsed {index}/{len(midi_files)} MIDI files, notes={len(notes)}")

        if len(notes) >= max_notes:
            break

    notes_array = np.asarray(notes, dtype=np.int16)
    np.save(cache_path, notes_array)
    return notes_array


def make_eval_batch(notes, seed_len, emotion_vector, sample_count, rng):
    start_min = (
        TRAIN_NOTE_CUTOFF
        if len(notes) > TRAIN_NOTE_CUTOFF + seed_len + 1
        else max(0, len(notes) // 2)
    )
    start_max = len(notes) - seed_len - 1
    if start_max <= start_min:
        raise ValueError(
            f"Not enough notes for held-out evaluation: have {len(notes)}, "
            f"need more than {start_min + seed_len + 1}."
        )

    indices = rng.integers(start_min, start_max, size=sample_count)
    x = np.empty((sample_count, seed_len + len(emotion_vector)), dtype=np.float32)
    y = np.empty((sample_count,), dtype=np.int64)

    for row, idx in enumerate(indices):
        x[row, :seed_len] = notes[idx : idx + seed_len]
        x[row, seed_len:] = emotion_vector
        y[row] = int(notes[idx + seed_len])

    return x, y, indices


def next_note_metrics(model, notes, seed_len, sample_count, rng):
    x, y, _ = make_eval_batch(
        notes, seed_len, EMOTION_MAP["relax"], sample_count, rng
    )
    start = time.perf_counter()
    probabilities = model.predict(x, batch_size=256, verbose=0)
    elapsed = time.perf_counter() - start
    probabilities = normalize(probabilities)

    target_prob = probabilities[np.arange(len(y)), y]
    nll = -np.log(np.maximum(target_prob, 1e-12))
    top_sorted = np.argsort(probabilities, axis=1)[:, -5:]

    return {
        "seed_len": seed_len,
        "samples": int(sample_count),
        "cross_entropy_nats": float(np.mean(nll)),
        "perplexity": float(np.exp(np.mean(nll))),
        "top1_accuracy": float(np.mean(top_sorted[:, -1] == y)),
        "top3_accuracy": float(np.mean([target in row[-3:] for target, row in zip(y, top_sorted)])),
        "top5_accuracy": float(np.mean([target in row for target, row in zip(y, top_sorted)])),
        "mean_entropy_nats": float(
            np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1))
        ),
        "examples_per_second": float(sample_count / max(elapsed, 1e-9)),
    }


def baseline_metrics(notes, seed_len, sample_count, rng):
    x, y, _ = make_eval_batch(notes, seed_len, EMOTION_MAP["relax"], sample_count, rng)
    train_notes = notes[: min(TRAIN_NOTE_CUTOFF, len(notes) - seed_len - 1)]
    counts = np.bincount(train_notes.astype(np.int64), minlength=128).astype(np.float64)
    unigram = counts / counts.sum()
    nll = -np.log(np.maximum(unigram[y], 1e-12))
    mode = int(np.argmax(unigram))

    previous_note_predictions = x[:, seed_len - 1].astype(np.int64)
    return {
        "unigram_cross_entropy_nats": float(np.mean(nll)),
        "unigram_perplexity": float(np.exp(np.mean(nll))),
        "unigram_mode_accuracy": float(np.mean(y == mode)),
        "previous_note_accuracy": float(np.mean(y == previous_note_predictions)),
    }


def normalize(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.maximum(probabilities, 1e-12)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def js_divergence(p, q):
    p = normalize(np.asarray([p]))[0]
    q = normalize(np.asarray([q]))[0]
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)))
    kl_qm = np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)))
    return 0.5 * (kl_pm + kl_qm)


def conditioning_sensitivity(model, notes, seed_len, sample_count, rng):
    start_min = (
        TRAIN_NOTE_CUTOFF
        if len(notes) > TRAIN_NOTE_CUTOFF + seed_len + 1
        else max(0, len(notes) // 2)
    )
    start_max = len(notes) - seed_len - 1
    indices = rng.integers(start_min, start_max, size=sample_count)

    batches = {}
    for emotion, vector in EMOTION_MAP.items():
        x = np.empty((sample_count, seed_len + len(vector)), dtype=np.float32)
        for row, idx in enumerate(indices):
            x[row, :seed_len] = notes[idx : idx + seed_len]
            x[row, seed_len:] = vector
        batches[emotion] = normalize(model.predict(x, batch_size=256, verbose=0))

    argmaxes = {emotion: np.argmax(preds, axis=1) for emotion, preds in batches.items()}
    unique_argmax_count = []
    pairwise_js = []
    identical_distribution_pairs = 0

    for row in range(sample_count):
        unique_argmax_count.append(
            len({int(argmaxes[emotion][row]) for emotion in EMOTION_MAP})
        )
        for left, right in combinations(EMOTION_MAP.keys(), 2):
            divergence = js_divergence(batches[left][row], batches[right][row])
            pairwise_js.append(divergence)
            if divergence < 1e-6:
                identical_distribution_pairs += 1

    return {
        "seed_len": seed_len,
        "samples": int(sample_count),
        "mean_unique_argmax_notes_across_emotions": float(np.mean(unique_argmax_count)),
        "share_all_emotions_same_argmax": float(np.mean(np.asarray(unique_argmax_count) == 1)),
        "mean_pairwise_js_divergence_nats": float(np.mean(pairwise_js)),
        "median_pairwise_js_divergence_nats": float(np.median(pairwise_js)),
        "share_nearly_identical_distribution_pairs": float(
            identical_distribution_pairs / max(1, len(pairwise_js))
        ),
    }


def sample_from_distribution(probabilities, rng, temperature):
    probabilities = np.maximum(np.asarray(probabilities, dtype=np.float64), 1e-12)
    logits = np.log(probabilities) / max(temperature, 0.05)
    exp_logits = np.exp(logits - np.max(logits))
    adjusted = exp_logits / exp_logits.sum()
    return int(rng.choice(len(adjusted), p=adjusted))


def fold_to_range(note, low, high):
    note = int(round(note))
    while note < low:
        note += 12
    while note > high:
        note -= 12
    return max(low, min(high, note))


def scale_candidates(emotion):
    config = EMOTION_CONFIG[emotion]
    low, high = config["range"]
    root = config["root"]
    candidates = []
    for octave in range(-3, 7):
        for degree in config["scale"]:
            candidate = root + octave * 12 + degree
            if low <= candidate <= high:
                candidates.append(candidate)
    return candidates


def quantize_to_scale(note, emotion):
    candidates = scale_candidates(emotion)
    return min(candidates, key=lambda candidate: abs(candidate - note))


def avoid_repetition(note, previous_note, emotion, repeat_count):
    if previous_note is None:
        return note

    config = EMOTION_CONFIG[emotion]
    low, high = config["range"]

    if note == previous_note and repeat_count >= 1:
        direction = 1 if previous_note < (low + high) / 2 else -1
        note = quantize_to_scale(previous_note + direction * 2, emotion)

    if abs(note - previous_note) > config["max_leap"]:
        while note - previous_note > config["max_leap"]:
            note -= 12
        while previous_note - note > config["max_leap"]:
            note += 12
        note = quantize_to_scale(fold_to_range(note, low, high), emotion)

    return max(low, min(high, note))


def shape_model_note(raw_note, previous_note, emotion, repeat_count):
    config = EMOTION_CONFIG[emotion]
    low, high = config["range"]
    note = fold_to_range(raw_note, low, high)
    note = quantize_to_scale(note, emotion)
    return int(max(MIN_MIDI, min(MAX_MIDI, avoid_repetition(note, previous_note, emotion, repeat_count))))


def in_scale(note, emotion):
    return int(note) in set(scale_candidates(emotion))


def summarize_generated(notes, emotion):
    notes = np.asarray(notes, dtype=np.int64)
    intervals = np.diff(notes)
    low, high = EMOTION_CONFIG[emotion]["range"]
    max_leap = EMOTION_CONFIG[emotion]["max_leap"]

    return {
        "note_count": int(len(notes)),
        "min_note": int(notes.min()),
        "max_note": int(notes.max()),
        "mean_note": float(notes.mean()),
        "unique_notes": int(len(set(notes.tolist()))),
        "consecutive_repeat_rate": float(np.mean(intervals == 0)) if len(intervals) else 0.0,
        "mean_abs_interval": float(np.mean(np.abs(intervals))) if len(intervals) else 0.0,
        "large_leap_rate": float(np.mean(np.abs(intervals) > max_leap)) if len(intervals) else 0.0,
        "range_compliance": float(np.mean((notes >= low) & (notes <= high))),
        "scale_compliance": float(np.mean([in_scale(note, emotion) for note in notes])),
    }


def generation_metrics(model, sequences_per_emotion, notes_per_sequence, rng):
    per_emotion = {}
    inference_times = []

    for emotion, config in EMOTION_CONFIG.items():
        raw_notes = []
        shaped_notes = []

        for offset in range(sequences_per_emotion):
            pattern = list(config["seed"])
            # Small deterministic rotations avoid every sample starting identically.
            pattern = pattern[offset % len(pattern) :] + pattern[: offset % len(pattern)]
            previous_note = pattern[-1]
            repeat_count = 0

            for _ in range(notes_per_sequence):
                x_input = np.asarray([pattern[-8:] + EMOTION_MAP[emotion]], dtype=np.float32)
                start = time.perf_counter()
                prediction = normalize(model(x_input, training=False).numpy())[0]
                inference_times.append(time.perf_counter() - start)

                raw_note = sample_from_distribution(
                    prediction, rng, config["temperature"]
                )
                shaped_note = shape_model_note(
                    raw_note, previous_note, emotion, repeat_count
                )

                raw_notes.append(raw_note)
                shaped_notes.append(shaped_note)

                if shaped_note == previous_note:
                    repeat_count += 1
                else:
                    repeat_count = 0

                pattern.append(shaped_note)
                pattern = pattern[-8:]
                previous_note = shaped_note

        per_emotion[emotion] = {
            "raw": summarize_generated(raw_notes, emotion),
            "shaped": summarize_generated(shaped_notes, emotion),
        }

    return {
        "per_emotion": per_emotion,
        "median_inference_ms_per_note": float(np.median(inference_times) * 1000),
        "mean_inference_ms_per_note": float(np.mean(inference_times) * 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-notes", type=int, default=1_100_000)
    parser.add_argument("--test-samples", type=int, default=4_000)
    parser.add_argument("--conditioning-samples", type=int, default=512)
    parser.add_argument("--sequences-per-emotion", type=int, default=12)
    parser.add_argument("--notes-per-sequence", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    rng = np.random.default_rng(args.seed)
    notes = extract_notes(args.max_notes)
    print(f"Loaded {len(notes)} MAESTRO notes for evaluation")

    model = load_model(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}, input_shape={model.input_shape}, output_shape={model.output_shape}")

    metrics = {
        "model_path": MODEL_PATH,
        "model_input_shape": str(model.input_shape),
        "model_output_shape": str(model.output_shape),
        "dataset": {
            "source": MAESTRO_URL,
            "notes_used": int(len(notes)),
            "training_note_cutoff_from_notebook": TRAIN_NOTE_CUTOFF,
            "heldout_region_start": int(
                TRAIN_NOTE_CUTOFF
                if len(notes) > TRAIN_NOTE_CUTOFF + 51
                else max(0, len(notes) // 2)
            ),
        },
        "next_note": {
            "training_shape_seed_50": next_note_metrics(
                model, notes, 50, args.test_samples, rng
            ),
            "deployed_shape_seed_8": next_note_metrics(
                model, notes, 8, args.test_samples, rng
            ),
            "baselines_seed_50": baseline_metrics(
                notes, 50, args.test_samples, rng
            ),
        },
        "conditioning_sensitivity": {
            "training_shape_seed_50": conditioning_sensitivity(
                model, notes, 50, args.conditioning_samples, rng
            ),
            "deployed_shape_seed_8": conditioning_sensitivity(
                model, notes, 8, args.conditioning_samples, rng
            ),
        },
        "generation_quality": generation_metrics(
            model, args.sequences_per_emotion, args.notes_per_sequence, rng
        ),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    output_path = ARTIFACT_DIR / "music_model_evaluation.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

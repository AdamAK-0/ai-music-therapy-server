# Music Generation Model Evaluation Report

Date: 2026-04-30

## Executive Summary

The current `music_model_emotion.h5` is usable as a lightweight next-note piano generator, but it should not be described as a fully emotion-trained model. On a held-out MAESTRO note stream, it beats simple baselines clearly: with the original 50-note training context it reaches 19.25% top-1 accuracy, 49.00% top-5 accuracy, and perplexity 26.34, compared with a unigram baseline perplexity of 59.99. In the deployed 8-note server context, quality drops to 14.03% top-1 accuracy and perplexity 36.29, which is expected because the notebook trained on 50 notes plus the emotion vector.

The four-emotion behavior in the app is valid as a deployed control system, but the validity comes mainly from the server using the detected emotion label to select a seed, pitch range, scale, sampling temperature, and maximum leap. The raw neural model was trained with only the `relax` emotion vector, so the vector was not learned from emotion-labeled music. Because of that, I did not replace the model with a `v2` in this pass: training a real `v2` needs an emotion-labeled symbolic music dataset such as EMOPIA/EMOPIA+, not pseudo-labels over MAESTRO. Promoting a pseudo-labeled `v2` as "better than previous work" would not be scientifically defensible.

## Artifacts Reviewed

- Training notebook: `Untitled80.ipynb`
- Deployed model: `music_model_emotion.h5`
- Runtime server: `app.py`
- Evaluation script added in this pass: `evaluate_music_model.py`
- Evaluation output: `evaluation_artifacts/music_model_evaluation.json`

`evaluation_artifacts/`, `maestro-v3.0.0/`, and `maestro-v3.0.0-midi.zip` are ignored in `.gitignore` so local test data and generated metrics are not pushed to GitHub.

## How The Current Model Was Trained

The notebook downloaded MAESTRO v3.0.0 MIDI data, extracted MIDI note pitches, and trained next-note prediction from a flat pitch sequence.

The first model used:

```text
Input: 50 MIDI pitches
Embedding(128, 100)
LSTM(256, return_sequences=True)
LSTM(256)
Dense(128, relu)
Dense(128, softmax)
Loss: sparse categorical crossentropy
Epochs: 20
Batch size: 64
```

The emotion model then changed the input to:

```text
50 MIDI pitches + 4 binary emotion-vector values = 54 input tokens
```

The vector mapping was:

```text
relax = [1, 0, 0, 0]
happy = [0, 1, 0, 0]
sad   = [0, 0, 1, 0]
focus = [0, 0, 0, 1]
```

Important limitation: the notebook built `X_new` with `get_emotion_vector("relax")` for every training sequence. That means the trained neural network only saw the `relax` vector during training. Also, because the first layer is an `Embedding`, the 0/1 vector values are treated as token IDs, not as a continuous semantic emotion vector.

## Why The Emotion Vector Is Still Useful

A one-hot emotion vector is a standard way to represent categorical control information in conditional generation, but it is valid only when the model is trained with examples across those categories.

For this project, the vector is valid in a narrower runtime sense:

- The frontend gets a stable label from the emotion detector: `relax`, `happy`, `sad`, or `focus`.
- The music server maps that label to a one-hot vector and passes it into the model.
- More importantly, the same label drives deterministic musical controls in `app.py`: seed pattern, pitch range, scale, temperature, and maximum allowed leap.
- The post-processing folds generated notes into the target range, quantizes them to the target scale, and suppresses harsh repetition or large leaps.

So the current system is emotion-controllable, but not because the neural model learned emotional music classes. It is emotion-controllable because the server wraps a next-note model with emotion-specific music-theory constraints.

## Evaluation Method

I added and ran:

```powershell
python evaluate_music_model.py --max-notes 1100000 --test-samples 4000 --conditioning-samples 512 --sequences-per-emotion 12 --notes-per-sequence 64
```

The script downloaded MAESTRO v3.0.0 MIDI if missing, cached the first 1,100,000 extracted note pitches, and evaluated on notes starting at index 1,000,000. This keeps the test region after the notebook's first-million-note training cutoff.

The evaluation covered:

- Next-note prediction with the original 50-note training shape.
- Next-note prediction with the deployed 8-note server shape.
- Unigram and previous-note baselines.
- Sensitivity of the raw neural distribution to the four emotion vectors.
- End-to-end generated-note quality after server shaping.
- Local CPU inference latency.

## Next-Note Prediction Results

| Model / baseline | Context | Cross-entropy | Perplexity | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Current model | 50 notes + vector | 3.271 nats | 26.34 | 19.25% | 37.15% | 49.00% |
| Current model | 8 notes + vector | 3.592 nats | 36.29 | 14.03% | 30.23% | 42.45% |
| Unigram baseline | none | 4.094 nats | 59.99 | 2.88% mode accuracy | n/a | n/a |
| Previous-note baseline | previous note only | n/a | n/a | 1.30% | n/a | n/a |

Interpretation:

- The model learned real pitch-sequence structure from MAESTRO.
- The deployed 8-note context is weaker than the 50-note training context, but still well above simple baselines.
- The server's 8-note chunking is acceptable for latency, but it sacrifices some predictive accuracy.

## Emotion-Conditioning Sensitivity

| Test | Mean unique top prediction across four emotions | Same top prediction for all emotions | Mean pairwise JS divergence |
|---|---:|---:|---:|
| 50-note input | 1.877 / 4 | 36.72% | 0.0677 nats |
| 8-note input | 1.955 / 4 | 33.59% | 0.0586 nats |

Interpretation:

- The raw model output changes somewhat when the emotion vector changes.
- However, because the model was only trained on the `relax` vector, this sensitivity is not evidence of learned happy/sad/focus semantics.
- The changes likely come from the LSTM seeing a different order of `0` and `1` tokens at the end of the sequence.

## Generated Music Quality

Each emotion generated 12 sequences of 64 notes. The table below reports the post-processed notes that the frontend would actually hear.

| Emotion | Mean MIDI note | Unique notes | Repeat rate | Mean abs interval | Large leap rate | Range compliance | Scale compliance |
|---|---:|---:|---:|---:|---:|---:|---:|
| relax | 61.77 | 13 | 11.60% | 3.38 | 1.56% | 100% | 100% |
| happy | 71.20 | 18 | 8.47% | 3.95 | 1.17% | 100% | 100% |
| sad | 61.41 | 18 | 9.65% | 3.24 | 5.35% | 100% | 100% |
| focus | 65.15 | 13 | 10.82% | 3.22 | 3.78% | 100% | 100% |

Raw model samples before shaping had 79.95%-87.63% range compliance, 62.63%-78.78% scale compliance, and 32.33%-54.76% large-leap rate depending on emotion. The server shaping is therefore doing a lot of useful work: it turns a rough next-note model into stable, playable, emotion-differentiated output.

## Latency

On the local Windows CPU environment:

- Median inference time: 81.93 ms per generated note.
- Mean inference time: 84.08 ms per generated note.
- An 8-note chunk is roughly 0.66-0.67 seconds of model compute before socket overhead.

This explains why 16-note chunks felt slow in the browser. The current 8-note chunk size is the better default for responsiveness.

## Comparison With Previous Literature

| Work | What it contributes | Comparison to this project |
|---|---|---|
| MAESTRO dataset, Google/Magenta | Large aligned piano MIDI/audio dataset, about 200 hours with close note/audio alignment. Source: https://magenta.withgoogle.com/datasets/maestro | Good fit for piano next-note training, but it does not provide emotion labels. Our model uses MAESTRO appropriately for pitch modeling, not for supervised emotion learning. |
| Performance RNN / This Time with Feeling | LSTM-based expressive performance generation that models notes, timing, and dynamics. Source: https://arxiv.org/abs/1808.03715 | Our model predicts pitch only. Emotion expression in therapy music would benefit from adding duration, velocity, and timing outputs. |
| Music Transformer | Relative-attention Transformer designed for longer-range musical structure. Source: https://arxiv.org/abs/1809.04281 | Our LSTM has a short context, especially in deployment with 8 notes. It is lighter and faster, but cannot claim the long-term structure strength of Transformer-based systems. |
| MuseNet / Sparse Transformer | Large-scale Transformer-style music generation over long token sequences. Source: https://openai.com/blog/musenet | Our model is far smaller and real-time friendly, but not comparable in scale, long-range conditioning, or stylistic coverage. |
| DeepBach | Steerable generation specialized for Bach chorales. Source: https://arxiv.org/abs/1612.01010 | DeepBach is polyphonic and style-specific. Our app is monophonic piano-note generation optimized for interactive therapy use. |
| C-RNN-GAN | Adversarial recurrent generation for continuous music sequences. Source: https://arxiv.org/abs/1611.09904 | Our model uses supervised next-note prediction, which is simpler to train and evaluate but less ambitious than adversarial sequence generation. |
| EMOPIA | Symbolic/audio pop-piano emotion dataset with 1,087 clips from 387 songs and four annotators. Source: https://arxiv.org/abs/2108.01374 | This is the kind of dataset needed for a real `v2` emotion-conditioned model. Our current model does not yet use it. |
| Russell valence-arousal affect model | A common two-dimensional emotion framework: valence and arousal. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC2367156/ | The four app categories can be understood roughly through valence/arousal, but `focus` is an application state more than a pure emotion label. |

## Should We Train A V2?

Yes, but not from the current MAESTRO-only setup if the goal is a scientifically valid emotion-conditioned model.

The current model is good enough to keep as `v1` because it is:

- Small: 953,090 total parameters.
- Fast enough for real-time 8-note chunks.
- Better than simple next-note baselines.
- Stable after server-side shaping.

But a true `v2` should be trained from emotion-labeled symbolic data. The safest route is:

1. Keep `music_model_emotion.h5` intact as `v1`.
2. Download EMOPIA or EMOPIA+ into an ignored dataset folder.
3. Map the dataset's emotion quadrants to the app categories carefully:
   - high valence / high arousal -> `happy`
   - low valence / low arousal -> `sad`
   - high valence / low arousal -> `relax`
   - high arousal / concentration-like or neutral/controlled clips -> `focus`, or rename this class if the data cannot support it cleanly
4. Train a model with separate inputs:
   - note sequence input through an embedding/LSTM or Transformer encoder
   - emotion input through a dense emotion embedding
   - concatenate emotion features into the recurrent state or each timestep, not as raw `0`/`1` tokens inside the pitch sequence
5. Predict more than pitch:
   - pitch
   - duration
   - velocity
   - optional rest/onset timing
6. Evaluate against:
   - next-token metrics on held-out clips
   - generated-note range/scale/leap metrics
   - emotion-classifier agreement on generated clips
   - human listening tests, even small ones
   - latency under the frontend's chunking budget

## Promotion Criteria For V2

I would only switch the server to `music_model_emotion_v2.h5` if it meets these gates:

- Top-5 next-pitch accuracy at least matches v1 on held-out data.
- Emotion-conditioned outputs are separable before post-processing, not only after range/scale shaping.
- Generated clips keep large-leap rate below 5% after shaping.
- Median per-note inference stays below 100 ms on local CPU or the deployment target.
- A generated-clip emotion classifier agrees with the requested emotion substantially above chance.
- Listening checks do not reveal stuck notes, pitch drift, or harsh repetition.

## Final Assessment

The current model is a reasonable v1 for the live app after the recent server improvements. It should be described as:

> A MAESTRO-trained next-note LSTM model wrapped with emotion-aware generation controls.

It should not be described as:

> A neural model that learned emotion-conditioned music generation.

The next serious research step is a separate `v2` trained and tested on an emotion-labeled symbolic dataset. Until then, the current architecture is acceptable for the product prototype because the emotion detector controls the generation label, and the music server reliably turns that label into different playable musical behaviors.

## Server2 Training Update - 2026-04-30

The separate V2 project now exists at:

```text
C:\Users\Adam\ai-music-therapy-server2
```

The detailed report for that project is:

```text
C:\Users\Adam\ai-music-therapy-server2\PROJECT_REPORT.md
```

The new best checkpoints are:

- `models/stream_v3/model.keras`
- `models/song_v2/model.keras`

The major change is that both new models use a larger two-stage training setup:

1. Pretrain on the AILabs symbolic corpus to learn musical motion.
2. Fine-tune on EMOPIA so emotion conditioning comes from real emotion labels.

The main metrics used are:

- Cross-entropy: the primary held-out validation loss. Lower means the model assigns higher probability to real unseen music.
- Perplexity: `exp(cross_entropy)`. Lower means fewer effective choices and less uncertainty.
- Top-1 accuracy: whether the most likely prediction exactly matches the held-out target.
- Top-5 accuracy: the most important accuracy metric for generation, because music has many valid continuations and the sampler uses ranked probabilities.
- Field-level accuracy: for `song_v2`, each musical field is evaluated separately: tempo, chord, bar-beat, type, pitch, duration, velocity, and emotion.
- Generation sanity metrics: note count, unique pitch ratio, pitch range, mean pitch, mean velocity, mean duration, immediate repeat rate, and mean interval.

`stream_v3` improved over `stream_v2`:

| Metric | stream_v2 | stream_v3 |
|---|---:|---:|
| Validation loss | 8.0371 | 7.8229 |
| Pitch perplexity | 23.77 | 19.59 |
| Pitch top-1 | 14.32% | 19.79% |
| Pitch top-5 | 47.47% | 53.20% |
| Duration top-5 | 77.15% | 77.73% |
| Velocity top-5 | 44.21% | 45.06% |

`song_v2` replaces the old giant event-token model with a factorized Transformer. This is more useful because one event has several independent musical fields, and a single compound-token mistake should not hide whether pitch, duration, velocity, tempo, or chord improved.

`song_v2` held-out EMOPIA results:

| Field | Top-1 | Top-5 | Perplexity |
|---|---:|---:|---:|
| tempo | 68.86% | 93.83% | 2.64 |
| chord | 71.92% | 95.67% | 2.17 |
| bar-beat | 68.42% | 93.88% | 2.76 |
| type | 76.60% | 100.00% | 1.55 |
| pitch | 42.52% | 67.43% | 10.08 |
| duration | 50.07% | 86.44% | 5.03 |
| velocity | 43.32% | 64.80% | 8.43 |
| emotion | 100.00% | 100.00% | 1.00 |

This identifies a real improvement because the new model is no longer only a next-pitch prototype. It learns structure, note type, pitch, timing, dynamics, and emotion fields together. It also generated a demo MIDI:

```text
C:\Users\Adam\ai-music-therapy-server2\generated\song_v2_happy.mid
```

That file contains 342 generated notes and lasts about 102.86 seconds.

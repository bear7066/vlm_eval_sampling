# VLM Eval Sampling

Standalone Hugging Face video sampling runner.

It reads one video, samples one frame every `N` seconds, sends the latest
`SEQUENCE_FRAMES` sampled frames to a VLM, and writes one JSONL record per
sample.

## Files

- `hf_eval_sampling.py`: standalone Python runner.
- `run.sh`: convenience wrapper.
- `requirements.txt`: Python dependencies for local Hugging Face inference.

## Setup

```bash
cd vlm_eval_sampling
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If the Hugging Face model is gated or private, set:

```bash
export HF_TOKEN=hf_xxx
```

## Run Full Model

Default model:

```bash
./run.sh /path/to/video.mp4
```

Equivalent explicit command:

```bash
MODEL=bear7011/gemma4-e4b-webvid4K_FT \
BASE_MODEL= \
./run.sh /path/to/video.mp4
```

The default output is:

```text
logs/<video>-bear7011_gemma4-e4b-webvid4K_FT-1fps.jsonl
```

## Run PEFT Adapter

For adapter-only checkpoints, set `BASE_MODEL`:

```bash
MODEL=bear7011/gemma4-e4b-kinetic3K_FT \
BASE_MODEL=google/gemma-4-e4b-it \
./run.sh /path/to/video.mp4
```

## Common Options

```bash
./run.sh /path/to/video.mp4 --max-frames 5
./run.sh /path/to/video.mp4 --duration-seconds 10
./run.sh /path/to/video.mp4 --resize-width 512
SAMPLE_EVERY_SECONDS=0.5 ./run.sh /path/to/video.mp4
OUTPUT=logs/test.jsonl ./run.sh /path/to/video.mp4
```

Dry run:

```bash
DRY_RUN=1 ./run.sh /path/to/video.mp4 --max-frames 1
```

## JSONL Schema

Each line contains:

- `timestamp`
- `backend`
- `video_path`
- `sample_index`
- `frame_index`
- `video_time_sec`
- `sequence_frame_indices`
- `sequence_times_sec`
- `image_size`
- `sampling`
- `video`
- `model`
- `base_model`
- `prompt`
- `text`
- `metrics`

## Notes

- The default model is a full Hugging Face model, so `BASE_MODEL` is empty.
- The runner uses the latest 4 sampled frames by default because the evaluation
  prompt is sequence-oriented.
- First run may download large model files from Hugging Face.

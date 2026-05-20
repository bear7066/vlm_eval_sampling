#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALL_DIR="$(pwd)"

load_env_defaults() {
  local env_file line key value
  for env_file in "$ROOT_DIR/.env" "$HOME/.env"; do
    [[ -f "$env_file" ]] || continue
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "$line" || "${line:0:1}" == "#" || "$line" != *=* ]] && continue

      key="${line%%=*}"
      value="${line#*=}"
      key="${key%"${key##*[![:space:]]}"}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"

      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
      [[ -v "$key" ]] && continue

      if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
      fi

      export "$key=$value"
    done < "$env_file"
  done
}

load_env_defaults

PATH="$HOME/.local/bin:$PATH"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [video_path] [extra hf_eval_sampling.py args...]

Examples:
  ./run.sh /path/to/input.mp4
  VIDEO=/path/to/input.mp4 ./run.sh --max-frames 5
  HF_TOKEN=hf_xxx ./run.sh /path/to/input.mp4

Full model default:
  MODEL=bear7011/gemma4-e4b-webvid4K_FT
  BASE_MODEL=

Adapter example:
  MODEL=bear7011/gemma4-e4b-kinetic3K_FT \
  BASE_MODEL=google/gemma-4-e4b-it \
  ./run.sh /path/to/input.mp4

Environment overrides:
  MODEL                  default: bear7011/gemma4-e4b-webvid4K_FT
  BASE_MODEL             default: empty; set for PEFT adapters
  OUTPUT                 default: logs/<video>-<model>-1fps.jsonl
  SAMPLE_EVERY_SECONDS   default: 1
  SEQUENCE_FRAMES        default: 4
  PROMPT                 default: sequence action prompt
  MAX_TOKENS             default: 128
  RESIZE_WIDTH           optional
  HF_TOKEN               optional, useful for gated/private models
  .env                   loads ROOT_DIR/.env then HOME/.env without overriding exported vars
  HF_DEVICE_MAP          default: auto
  HF_TORCH_DTYPE         default: auto
  PYTHON                 optional override; defaults to uv run
  DRY_RUN                set to 1 to print command only
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  VIDEO="$1"
  shift
else
  VIDEO="${VIDEO:-}"
fi

if [[ -z "$VIDEO" ]]; then
  usage
  echo "error=missing video path" >&2
  exit 2
fi

if [[ "$VIDEO" != /* ]]; then
  if [[ -e "$CALL_DIR/$VIDEO" ]]; then
    VIDEO="$CALL_DIR/$VIDEO"
  else
    VIDEO="$ROOT_DIR/$VIDEO"
  fi
fi

MODEL="${MODEL-bear7011/gemma3-4b-kinetic3K_FT}"
BASE_MODEL="${BASE_MODEL-}"
SAMPLE_EVERY_SECONDS="${SAMPLE_EVERY_SECONDS:-1}"
SEQUENCE_FRAMES="${SEQUENCE_FRAMES:-4}"
DEFAULT_PROMPT="Describe the main action briefly in 2~6 words."
# DEFAULT_PROMPT+=" What specific action or event is happening?"
PROMPT="${PROMPT:-$DEFAULT_PROMPT}"
MAX_TOKENS="${MAX_TOKENS:-512}"

VIDEO_NAME="$(basename "$VIDEO")"
VIDEO_STEM="${VIDEO_NAME%.*}"
MODEL_TAG="$(printf '%s' "$MODEL" | tr '/:' '__')"
OUTPUT="${OUTPUT:-$ROOT_DIR/logs/${VIDEO_STEM}-${MODEL_TAG}-1fps.jsonl}"
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$CALL_DIR/$OUTPUT"
fi

if [[ -n "${PYTHON:-}" ]]; then
  CMD=(
    "$PYTHON"
    "$ROOT_DIR/hf_eval_sampling.py"
  )
elif command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
  CMD=(
    "$UV_BIN"
    run
    --with-requirements
    "$ROOT_DIR/requirements.txt"
    python
    "$ROOT_DIR/hf_eval_sampling.py"
  )
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  CMD=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/hf_eval_sampling.py"
  )
else
  echo "error=uv not found and $ROOT_DIR/.venv/bin/python is unavailable" >&2
  echo "hint=install uv, add it to PATH, or set PYTHON=/path/to/python" >&2
  exit 127
fi

CMD+=(
  "$VIDEO"
  "--output"
  "$OUTPUT"
  "--model"
  "$MODEL"
  "--base-model"
  "$BASE_MODEL"
  "--sample-every-seconds"
  "$SAMPLE_EVERY_SECONDS"
  "--sequence-frames"
  "$SEQUENCE_FRAMES"
  "--prompt"
  "$PROMPT"
  "--max-tokens"
  "$MAX_TOKENS"
)

if [[ -n "${RESIZE_WIDTH:-}" ]]; then
  CMD+=("--resize-width" "$RESIZE_WIDTH")
fi

CMD+=("$@")

mkdir -p "$(dirname "$OUTPUT")"

echo "video=$VIDEO"
echo "model=$MODEL"
echo "base_model=$BASE_MODEL"
echo "backend=huggingface"
echo "sample_every_seconds=$SAMPLE_EVERY_SECONDS"
echo "sequence_frames=$SEQUENCE_FRAMES"
echo "output=$OUTPUT"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "hf_token=present"
else
  echo "hf_token=missing"
fi

cd "$ROOT_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf "%q " "${CMD[@]}"
  printf "\n"
  exit 0
fi

exec "${CMD[@]}"

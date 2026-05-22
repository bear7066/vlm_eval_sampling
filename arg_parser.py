import argparse
import os
from helper import positive_int, positive_float, non_negative_float

DEFAULT_BASE_MODEL = ""
DEFAULT_MODEL = "bear7011/gemma4-e4b-webvid4K_FT"
DEFAULT_PROMPT = (
    "Describe the main action briefly in 2~6 words."
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hf-eval-sampling",
        description=(
            "Sample frames from a video, run a Hugging Face VLM locally, "
            "and write JSONL logs."
        ),
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-o",
        "--output",
        default="logs/hf-eval-sampling.jsonl",
        help="JSONL output path (default: logs/hf-eval-sampling.jsonl)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id or PEFT adapter id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=(
            "Hugging Face base model id for PEFT adapters. Leave empty when "
            "--model is a full Hugging Face model."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent with every sample")
    parser.add_argument(
        "--max-tokens",
        "--max-new-tokens",
        dest="max_new_tokens",
        type=positive_int,
        default=128,
        help="Maximum generated tokens (default: 128)",
    )
    parser.add_argument(
        "--sequence-frames",
        type=positive_int,
        default=4,
        help="Number of recent sampled frames to send per inference (default: 4)",
    )
    parser.add_argument(
        "--process-every",
        type=positive_int,
        default=None,
        help="Sample every N decoded video frames",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=positive_float,
        default=1.0,
        help="Sample one frame every N seconds when --process-every is not set (default: 1.0)",
    )
    parser.add_argument(
        "--start-seconds",
        type=non_negative_float,
        default=0.0,
        help="Start sampling at this video timestamp",
    )
    parser.add_argument(
        "--duration-seconds",
        type=positive_float,
        default=None,
        help="Only sample this many seconds from --start-seconds",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="Stop after this many sampled frames",
    )
    parser.add_argument(
        "--resize-width",
        type=positive_int,
        default=None,
        help="Resize sampled frames to this width before sending to the VLM",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output log instead of replacing it",
    )
    parser.add_argument(
        "--torch-dtype",
        default=os.environ.get("HF_TORCH_DTYPE", "auto"),
        help="torch dtype passed to from_pretrained (default: auto)",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("HF_DEVICE_MAP", "auto"),
        help="device_map passed to from_pretrained; set empty string to disable (default: auto)",
    )
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("HF_ATTN_IMPLEMENTATION"),
        help="Optional attention implementation, for example flash_attention_2",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load files already present in the local Hugging Face cache",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable info logs",
    )
    return parser

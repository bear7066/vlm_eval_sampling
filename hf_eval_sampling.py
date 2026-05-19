# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Offline video-to-Hugging Face VLM test runner.

Reads a video file, samples frames every N seconds, sends a short sequence of
recent frames to a Hugging Face PEFT adapter, and writes JSONL observations.
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

DEFAULT_BASE_MODEL = ""
DEFAULT_MODEL = "bear7011/gemma4-e4b-webvid4K_FT"
DEFAULT_PROMPT = (
    "Describe the main action briefly in 2~6 words."
)


def emit(message: str) -> None:
    print(message, flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


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


def get_video_info(capture: cv2.VideoCapture) -> dict[str, float | int]:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def resolve_sample_interval(args: argparse.Namespace, fps: float) -> tuple[int, float]:
    if args.process_every is not None:
        interval_frames = args.process_every
        interval_seconds = interval_frames / fps if fps > 0 else 0.0
        return interval_frames, interval_seconds

    if fps <= 0:
        raise ValueError("Video FPS is unavailable; set --process-every explicitly.")

    interval_frames = max(1, int(round(fps * args.sample_every_seconds)))
    interval_seconds = interval_frames / fps
    return interval_frames, interval_seconds


def frame_to_image(frame: Any, resize_width: int | None) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    if resize_width and image.width != resize_width:
        resize_height = max(1, int(round(image.height * (resize_width / image.width))))
        image = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
    return image


def build_recent_sequence(
    history: list[tuple[int, float, Image.Image]], sequence_frames: int
) -> tuple[list[Image.Image], list[int], list[float]]:
    selected = history[-sequence_frames:]
    if len(selected) < sequence_frames:
        selected = [selected[0]] * (sequence_frames - len(selected)) + selected
    return (
        [image for _, _, image in selected],
        [frame_index for frame_index, _, _ in selected],
        [video_time_sec for _, video_time_sec, _ in selected],
    )


def load_hf_components(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Hugging Face runtime dependency. Install with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    full_model = not args.base_model

    loader_kwargs: dict[str, Any] = {
        "torch_dtype": args.torch_dtype,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        loader_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        loader_kwargs["attn_implementation"] = args.attn_implementation
    if args.hf_token:
        loader_kwargs["token"] = args.hf_token

    if full_model:
        try:
            from transformers import AutoModelForImageTextToText as FullModelClass
        except ImportError:
            try:
                from transformers import Gemma4ForConditionalGeneration as FullModelClass
            except ImportError as exc:
                raise RuntimeError(
                    "Your transformers version does not provide a Gemma 4 image-text model "
                    "loader. Install/update with: python3 -m pip install -r requirements.txt"
                ) from exc

        emit(f"loading_full_model={args.model}")
        model = FullModelClass.from_pretrained(args.model, **loader_kwargs)
        processor_model = args.model
    else:
        try:
            from peft import PeftModel
            from transformers import Gemma4ForConditionalGeneration as BaseModelClass
        except ImportError as exc:
            raise RuntimeError(
                "Your Hugging Face runtime cannot load the PEFT Gemma 4 adapter. "
                "Install/update with: python3 -m pip install -r requirements.txt"
            ) from exc

        emit(f"loading_base_model={args.base_model}")
        base_model = BaseModelClass.from_pretrained(args.base_model, **loader_kwargs)

        adapter_kwargs: dict[str, Any] = {
            "local_files_only": args.local_files_only,
        }
        if args.hf_token:
            adapter_kwargs["token"] = args.hf_token

        emit(f"loading_adapter={args.model}")
        model = PeftModel.from_pretrained(base_model, args.model, **adapter_kwargs)
        processor_model = args.base_model

    model.eval()

    processor_kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.hf_token:
        processor_kwargs["token"] = args.hf_token
    processor = AutoProcessor.from_pretrained(processor_model, **processor_kwargs)

    return model, processor, torch


def get_input_device(model: Any, torch: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_inputs_to_device(inputs: Any, device: Any, torch: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def prepare_inputs(processor: Any, images: list[Image.Image], prompt: str) -> Any:
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return processor(text=[text], images=images, return_tensors="pt")


def decode_generated_text(processor: Any, generated_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def analyze_images(
    model: Any,
    processor: Any,
    torch: Any,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    start_time = time.perf_counter()
    inputs = prepare_inputs(processor, images, prompt)
    inputs = move_inputs_to_device(inputs, get_input_device(model, torch), torch)
    input_token_count = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids = output_ids[:, input_token_count:] if input_token_count else output_ids
    text = decode_generated_text(processor, generated_ids)
    inference_time = time.perf_counter() - start_time
    return text, inference_time


def make_log_record(
    args: argparse.Namespace,
    video_path: Path,
    video_info: dict[str, float | int],
    interval_frames: int,
    interval_seconds: float,
    sample_index: int,
    frame_index: int,
    video_time_sec: float,
    sequence_frame_indices: list[int],
    sequence_times_sec: list[float],
    image: Image.Image,
    text: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": "huggingface",
        "video_path": str(video_path),
        "sample_index": sample_index,
        "frame_index": frame_index,
        "video_time_sec": video_time_sec,
        "sequence_frame_indices": sequence_frame_indices,
        "sequence_times_sec": sequence_times_sec,
        "image_size": {"width": image.width, "height": image.height},
        "sampling": {
            "process_every_frames": interval_frames,
            "sample_every_seconds": interval_seconds,
            "sequence_frames": args.sequence_frames,
            "start_seconds": args.start_seconds,
            "duration_seconds": args.duration_seconds,
            "max_frames": args.max_frames,
        },
        "video": video_info,
        "model": args.model,
        "base_model": args.base_model,
        "prompt": args.prompt,
        "text": text,
        "metrics": metrics,
    }


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        video_info = get_video_info(capture)
        fps = float(video_info["fps"])
        interval_frames, interval_seconds = resolve_sample_interval(args, fps)

        start_frame = int(math.floor(args.start_seconds * fps)) if fps > 0 else 0
        end_frame = None
        if args.duration_seconds is not None and fps > 0:
            end_frame = start_frame + int(math.ceil(args.duration_seconds * fps))

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"

        emit(
            "video="
            f"{video_path} fps={fps:.3f} frames={video_info['frame_count']} "
            f"duration={video_info['duration_sec']:.3f}s "
            f"size={video_info['width']}x{video_info['height']}"
        )
        emit(
            "sampling="
            f"every {interval_frames} frame(s)"
            f" (~{interval_seconds:.3f}s), sequence_frames={args.sequence_frames}, "
            f"start={args.start_seconds:.3f}s, "
            f"duration={args.duration_seconds if args.duration_seconds is not None else 'full'}"
        )
        emit(f"output={output_path}")

        model, processor, torch = load_hf_components(args)

        sample_count = 0
        error_count = 0
        total_inference_time = 0.0
        history: list[tuple[int, float, Image.Image]] = []

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_index = start_frame

        with output_path.open(mode, encoding="utf-8") as log_file:
            while True:
                if end_frame is not None and frame_index >= end_frame:
                    break
                if args.max_frames is not None and sample_count >= args.max_frames:
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                if (frame_index - start_frame) % interval_frames == 0:
                    video_time_sec = frame_index / fps if fps > 0 else 0.0
                    image = frame_to_image(frame, args.resize_width)
                    history.append((frame_index, video_time_sec, image))
                    sequence_images, sequence_frame_indices, sequence_times_sec = (
                        build_recent_sequence(history, args.sequence_frames)
                    )

                    try:
                        text, inference_time = analyze_images(
                            model=model,
                            processor=processor,
                            torch=torch,
                            images=sequence_images,
                            prompt=args.prompt,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as exc:
                        text = f"Error: {exc}"
                        inference_time = 0.0
                        error_count += 1

                    sample_count += 1
                    total_inference_time += inference_time
                    avg_inference_time = total_inference_time / sample_count
                    metrics = {
                        "last_latency_ms": inference_time * 1000,
                        "avg_latency_ms": avg_inference_time * 1000,
                        "total_inferences": sample_count,
                        "is_processing": False,
                    }

                    record = make_log_record(
                        args=args,
                        video_path=video_path,
                        video_info=video_info,
                        interval_frames=interval_frames,
                        interval_seconds=interval_seconds,
                        sample_index=sample_count,
                        frame_index=frame_index,
                        video_time_sec=video_time_sec,
                        sequence_frame_indices=sequence_frame_indices,
                        sequence_times_sec=sequence_times_sec,
                        image=image,
                        text=text,
                        metrics=metrics,
                    )
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    emit(
                        f"[{sample_count}] frame={frame_index} "
                        f"t={video_time_sec:.3f}s latency={metrics['last_latency_ms']:.0f}ms "
                        f"text={text}"
                    )

                frame_index += 1

        final_avg_ms = (total_inference_time / sample_count * 1000) if sample_count else 0.0
        emit(
            "done "
            f"samples={sample_count} errors={error_count} avg_latency={final_avg_ms:.0f}ms "
            f"log={output_path}"
        )
        return 1 if error_count else 0
    finally:
        capture.release()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, force=True)
    logging.getLogger().setLevel(log_level)
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        if args.verbose:
            logging.exception("HF live VLM test failed")
        emit(f"error={exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

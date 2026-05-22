import argparse
import time, math, cv2
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image


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


def decode_generated_text(processor: Any, generated_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


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
        # "backend": "huggingface",
        # "video_path": str(video_path),
        # "sample_index": sample_index,
        # "frame_index": frame_index,
        # "video_time_sec": video_time_sec,
        # "sequence_frame_indices": sequence_frame_indices,
        # "sequence_times_sec": sequence_times_sec,
        # "image_size": {"width": image.width, "height": image.height},
        # "sampling": {
        #     "process_every_frames": interval_frames,
        #     "sample_every_seconds": interval_seconds,
        #     "sequence_frames": args.sequence_frames,
        #     "start_seconds": args.start_seconds,
        #     "duration_seconds": args.duration_seconds,
        #     "max_frames": args.max_frames,
        # },
        "process_every_frames": interval_seconds,
        "video": video_info,
        "model": args.model,
        "base_model": args.base_model,
        "prompt": args.prompt,
        "text": text,
        "metrics": metrics,
    }

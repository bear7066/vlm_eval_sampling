"""
Offline video-to-Hugging Face VLM test runner.

Reads a video file, samples frames every N seconds, sends a short sequence of
recent frames to a Hugging Face PEFT adapter, and writes JSONL observations.
"""

import argparse, json, logging, math, os, time, cv2, torch
from pathlib import Path
from typing import Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText as FullModelClass
from peft import PeftModel
from transformers import Gemma4ForConditionalGeneration as BaseModelClass


from arg_parser import build_parser
from helper import (
    build_recent_sequence,
    decode_generated_text,
    emit,
    frame_to_image,
    get_input_device,
    get_video_info,
    make_log_record,
    move_inputs_to_device,
    resolve_sample_interval,
)


def load_hf_components(args: argparse.Namespace) -> tuple[Any, Any, Any]:
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
        emit(f"loading_full_model={args.model}")
        model = FullModelClass.from_pretrained(args.model, **loader_kwargs)
        processor_model = args.model
    else: 
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

# VLM Eval Sampling

Standalone Hugging Face video sampling runner.

It reads one video, samples one frame every `N` seconds, sends the latest
`SEQUENCE_FRAMES` sampled frames to a VLM, and writes one JSONL record per
sample.


## Usage

1. 
```
uv sync
```

2. 
```
uv run main.py
```

## Parameters

| 參數 | 說明 | 預設值 |
| :--- | :--- | :--- |
| `video` | **Necessary**| - |
| `--model` | Hugging Face model or ID。 | `bear7011/gemma4-e4b-webvid4K_FT` |
| `--prompt` | X | `Describe the main action briefly in 2~6 words.` |
| `--sample-every-seconds` | X| `1.0` |
| `--sequence-frames` | 每次推論所包含的最近取樣影格數。 | `4` |
| `--output`, `-o` | X | `logs/hf-eval-sampling.jsonl` |
| `--start-seconds` | X | `0.0` |
| `--duration-seconds` | 取樣持續的總秒數。 | 全部 |
| `--trust-remote-code` | 是否信任遠端程式碼（載入特定模型時需要）。 | `false` |

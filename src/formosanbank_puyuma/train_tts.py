from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import load_jsonl, materialize_ljspeech_dataset, normalize_text, write_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and optionally launch TTS training")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="puyuma-tts")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--model", type=str, default="glow_tts")
    parser.add_argument("--text-cleaner", type=str, default="basic_cleaners")
    parser.add_argument("--use-phonemes", action="store_true")
    parser.add_argument("--phoneme-language", type=str, default="en-us")
    parser.add_argument("--allow-copy", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--restore-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_jsonl(args.manifest)
    if not rows:
        raise SystemExit("TTS manifest is empty")

    dataset_dir = args.output_dir / args.run_name
    metadata_path = materialize_ljspeech_dataset(
        rows,
        audio_root=args.audio_root,
        output_dir=dataset_dir,
        allow_copy=args.allow_copy,
    )
    config_path = dataset_dir / "config.json"
    config = _build_config(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir / "runs",
        run_name=args.run_name,
        sample_rate=args.sample_rate,
        epochs=args.epochs,
        model=args.model,
        text_cleaner=args.text_cleaner,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language,
    )
    write_text(config_path, json.dumps(config, ensure_ascii=False, indent=2))
    _write_run_notes(dataset_dir, metadata_path, config_path, args)

    if args.launch:
        if args.restore_path is None:
            raise SystemExit("--restore-path is required when --launch is set")
        try:
            import subprocess
            import sys
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("subprocess is unavailable") from exc

        command = [
            sys.executable,
            "-m",
            "TTS.bin.train_tts",
            "--config_path",
            str(config_path),
            "--restore_path",
            str(args.restore_path),
        ]
        subprocess.run(command, check=True)
    else:
        print(json.dumps({"dataset_dir": str(dataset_dir), "config_path": str(config_path)}, ensure_ascii=False, indent=2))

    return 0


def _build_config(
    *,
    dataset_dir: Path,
    output_dir: Path,
    run_name: str,
    sample_rate: int,
    epochs: int,
    model: str,
    text_cleaner: str,
    use_phonemes: bool,
    phoneme_language: str,
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "output_path": str(output_dir / run_name),
        "model": model,
        "batch_size": 32,
        "eval_batch_size": 16,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 2,
        "run_eval": True,
        "test_delay_epochs": -1,
        "epochs": epochs,
        "text_cleaner": text_cleaner,
        "use_phonemes": use_phonemes,
        "phoneme_language": phoneme_language,
        "phoneme_cache_path": "phoneme_cache",
        "print_step": 25,
        "print_eval": True,
        "mixed_precision": False,
        "audio": {"sample_rate": sample_rate},
        "datasets": [
            {
                "formatter": "ljspeech",
                "meta_file_train": "metadata.csv",
                "meta_file_val": "metadata.csv",
                "path": str(dataset_dir),
            }
        ],
        "test_sentences": [
            "semavalran.",
            "mavangavang u kinadamanan?",
            "adi, aku kamavangavangan.",
        ],
    }


def _write_run_notes(dataset_dir: Path, metadata_path: Path, config_path: Path, args: argparse.Namespace) -> None:
    notes = [
        "# TTS training scaffold",
        "",
        f"- dataset_dir: `{dataset_dir}`",
        f"- metadata_path: `{metadata_path}`",
        f"- config_path: `{config_path}`",
        f"- restore_path: `{args.restore_path}`",
        "",
        "Recommended launch command:",
        "",
        "```powershell",
        f"python -m TTS.bin.train_tts --config_path \"{config_path}\" --restore_path \"<pretrained-model>\"",
        "```",
        "",
        "If the chosen Coqui version requires different config keys, adjust `config.json` before launch.",
    ]
    write_text(dataset_dir / "README.md", "\n".join(notes) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

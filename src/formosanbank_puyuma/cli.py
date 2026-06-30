from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    compute_summary,
    dedupe_records,
    download_sources,
    format_summary_markdown,
    load_jsonl,
    load_records_from_xml_root,
    materialize_ljspeech_dataset,
    normalize_text,
    parse_xml_file,
    split_records,
    split_rows,
    source_audio_url,
    source_xml_url,
    supported_sources,
    tts_dedupe_key,
    tts_row,
    translation_dedupe_key,
    translation_row,
    write_csv,
    write_jsonl,
    write_text,
)
from .sources import CorpusSource


def _row_text(value: object | None) -> str:
    if value is None:
        return ""
    return normalize_text(str(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="formosanbank-puyuma")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-sources", help="List supported XML/audio sources")
    p.set_defaults(func=cmd_list_sources)

    p = sub.add_parser("download", help="Download XML and optionally audio")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--download-audio", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="Download, parse, dedupe, and write manifests")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--download-audio", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--no-dedupe", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("extract", help="Parse downloaded XML into manifests")
    p.add_argument("--xml-dir", type=Path, default=Path("data/raw/xml"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--no-dedupe", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("stats", help="Print a corpus summary")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--markdown", type=Path, default=None)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("split", help="Split a JSONL manifest")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--mode", choices=["translation", "tts"], default="translation")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("prepare-tts", help="Build an LJSpeech-style dataset folder")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-copy", action="store_true")
    p.set_defaults(func=cmd_prepare_tts)

    return parser


def cmd_list_sources(args: argparse.Namespace) -> int:
    for source in supported_sources():
        print(f"{source.source_key}\t{source.xml_relative_path}\t{source.audio_repo}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    data_dir = args.data_dir
    xml_root = data_dir / "raw" / "xml"
    audio_root = data_dir / "raw" / "audio"
    records = download_sources(
        supported_sources(),
        xml_root=xml_root,
        audio_root=audio_root,
        download_audio=args.download_audio,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps({"downloaded_records": len(records)}, ensure_ascii=False, indent=2))
    return 0


def _prepare_manifest_rows(records, *, no_dedupe: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    translation_records = list(records)
    tts_records = list(records)
    if not no_dedupe:
        translation_records, _ = dedupe_records(translation_records, key_fn=translation_dedupe_key)
        tts_records, _ = dedupe_records(tts_records, key_fn=tts_dedupe_key)
    translation_rows = [translation_row(record) for record in translation_records if record.target_text and record.source_text]
    tts_rows = [tts_row(record) for record in tts_records if record.audio_file and record.source_text]
    return translation_rows, tts_rows


def cmd_prepare(args: argparse.Namespace) -> int:
    data_dir = args.data_dir
    xml_root = data_dir / "raw" / "xml"
    audio_root = data_dir / "raw" / "audio"
    processed_dir = data_dir / "processed"
    records = download_sources(
        supported_sources(),
        xml_root=xml_root,
        audio_root=audio_root,
        download_audio=args.download_audio,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    summary = compute_summary(records)
    write_text(processed_dir / "summary.md", format_summary_markdown(summary))
    write_jsonl([record.to_dict() for record in records], processed_dir / "puyuma_examples.jsonl")
    translation_rows, tts_rows = _prepare_manifest_rows(records, no_dedupe=args.no_dedupe)
    split_translation = split_rows(
        translation_rows,
        key_fn=lambda row: _row_text(row.get("source_text")) + "\u241f" + _row_text(row.get("target_text")),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    split_tts = split_rows(
        tts_rows,
        key_fn=lambda row: _row_text(row.get("audio_file")) or _row_text(row.get("record_id")),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    _write_split_manifests(split_translation, processed_dir, prefix="translation")
    _write_split_manifests(split_tts, processed_dir, prefix="tts")
    write_jsonl(translation_rows, processed_dir / "translation_all.jsonl")
    write_jsonl(tts_rows, processed_dir / "tts_all.jsonl")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    records = load_records_from_xml_root(args.xml_dir)
    processed_dir = args.output_dir
    write_jsonl([record.to_dict() for record in records], processed_dir / "puyuma_examples.jsonl")
    translation_rows, tts_rows = _prepare_manifest_rows(records, no_dedupe=args.no_dedupe)
    split_translation = split_rows(
        translation_rows,
        key_fn=lambda row: _row_text(row.get("source_text")) + "\u241f" + _row_text(row.get("target_text")),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    split_tts = split_rows(
        tts_rows,
        key_fn=lambda row: _row_text(row.get("audio_file")) or _row_text(row.get("record_id")),
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    _write_split_manifests(split_translation, processed_dir, prefix="translation")
    _write_split_manifests(split_tts, processed_dir, prefix="tts")
    write_jsonl(translation_rows, processed_dir / "translation_all.jsonl")
    write_jsonl(tts_rows, processed_dir / "tts_all.jsonl")
    summary = compute_summary(records)
    write_text(processed_dir / "summary.md", format_summary_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    summary = {
        "total_records": len(rows),
        "by_group": {},
        "by_corpus": {},
        "by_dialect": {},
        "by_audio_mode": {},
        "missing_audio": 0,
        "missing_translation": 0,
        "total_duration_seconds": 0,
    }
    counters = {
        "by_group": {},
        "by_corpus": {},
        "by_dialect": {},
        "by_audio_mode": {},
    }
    for row in rows:
        group = _row_text(row.get("group", ""))
        corpus_slug = _row_text(row.get("corpus_slug", ""))
        dialect = _row_text(row.get("dialect", ""))
        audio_mode = _row_text(row.get("audio_mode", ""))
        counters["by_group"][group] = counters["by_group"].get(group, 0) + 1
        counters["by_corpus"][f"{group}/{corpus_slug}"] = counters["by_corpus"].get(
            f"{group}/{corpus_slug}",
            0,
        ) + 1
        counters["by_dialect"][dialect] = counters["by_dialect"].get(dialect, 0) + 1
        counters["by_audio_mode"][audio_mode] = counters["by_audio_mode"].get(audio_mode, 0) + 1
        if not row.get("audio_file"):
            summary["missing_audio"] += 1
        if not row.get("translation_zh") and not row.get("target_text"):
            summary["missing_translation"] += 1
        duration = row.get("duration_seconds")
        if isinstance(duration, (int, float)):
            summary["total_duration_seconds"] += duration
    summary.update(counters)
    if args.markdown:
        write_text(args.markdown, format_summary_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    if args.mode == "translation":
        split = split_rows(
            rows,
            key_fn=lambda row: _row_text(row.get("source_text")) + "\u241f" + _row_text(row.get("target_text")),
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )
    else:
        split = split_rows(
            rows,
            key_fn=lambda row: _row_text(row.get("audio_file")) or _row_text(row.get("record_id")),
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_rows_value in split.items():
        write_jsonl(split_rows_value, args.output_dir / f"{args.mode}_{split_name}.jsonl")
    print(json.dumps({name: len(rows) for name, rows in split.items()}, ensure_ascii=False, indent=2))
    return 0


def cmd_prepare_tts(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.input)
    materialize_ljspeech_dataset(
        rows,
        audio_root=args.audio_root,
        output_dir=args.output_dir,
        allow_copy=args.allow_copy,
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


def _write_split_manifests(split_map: dict[str, list[dict[str, object]]], output_dir: Path, *, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_map.items():
        write_jsonl(rows, output_dir / f"{prefix}_{split_name}.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

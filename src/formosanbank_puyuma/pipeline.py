from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from .sources import (
    CorpusSource,
    GITHUB_RAW_BASE,
    HF_DATASET_BASE,
    XML_LANG_ATTR,
    supported_sources,
)

USER_AGENT = "Codex-FormosanBank-Puyuma/0.1"


@dataclass(frozen=True, slots=True)
class ExampleRecord:
    record_id: str
    source_key: str
    group: str
    corpus_slug: str
    dialect: str
    language_code: str
    audio_mode: str
    xml_source_url: str
    xml_path: str
    citation: str
    copyright: str
    bibtex_citation: str
    root_source_label: str
    root_glottocode: str
    sentence_id: str
    original_form: str
    standard_form: str
    original_phonetic: str
    standard_phonetic: str
    translation_zh: str
    audio_file: str
    audio_url: str
    audio_start: float | None
    audio_end: float | None

    @property
    def source_text(self) -> str:
        return self.standard_form or self.original_form

    @property
    def source_phonetic(self) -> str:
        return self.standard_phonetic or self.original_phonetic

    @property
    def target_text(self) -> str:
        return self.translation_zh

    @property
    def duration_seconds(self) -> float | None:
        if self.audio_start is None or self.audio_end is None:
            return None
        return max(0.0, self.audio_end - self.audio_start)

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["source_text"] = self.source_text
        row["source_phonetic"] = self.source_phonetic
        row["target_text"] = self.target_text
        row["duration_seconds"] = self.duration_seconds
        return row


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def source_xml_url(source: CorpusSource) -> str:
    return source.xml_url


def source_audio_url(source: CorpusSource, audio_file: str) -> str:
    quoted = urllib.parse.quote(audio_file)
    if source.group == "ePark":
        return f"{HF_DATASET_BASE}/{source.audio_repo}/resolve/main/Puyuma/{source.dialect}_Puyuma/{quoted}"
    return f"{HF_DATASET_BASE}/{source.audio_repo}/resolve/main/{quoted}"


def local_xml_path(xml_root: Path, source: CorpusSource) -> Path:
    return xml_root / source.xml_relative_path


def local_audio_dir(audio_root: Path, source: CorpusSource) -> Path:
    if source.group == "ePark":
        return audio_root / source.audio_repo_slug / "Puyuma" / f"{source.dialect}_Puyuma"
    return audio_root / source.audio_repo_slug


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, *, overwrite: bool = False, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return False

    ensure_parent(dest)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(dest.parent), suffix=".tmp") as tmp:
                tmp_path = Path(tmp.name)
                shutil.copyfileobj(response, tmp)
        os.replace(tmp_path, dest)
        return True
    except Exception as exc:  # pragma: no cover - network failures are environment dependent
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} -> {dest}: {exc}") from exc


def download_sources(
    sources: Sequence[CorpusSource],
    *,
    xml_root: Path,
    audio_root: Path | None = None,
    download_audio: bool = False,
    overwrite: bool = False,
    limit: int | None = None,
) -> list[ExampleRecord]:
    downloaded_records: list[ExampleRecord] = []

    for index, source in enumerate(sources):
        if limit is not None and index >= limit:
            break

        xml_dest = local_xml_path(xml_root, source)
        download_file(source_xml_url(source), xml_dest, overwrite=overwrite)
        records = parse_xml_file(xml_dest, source)
        downloaded_records.extend(records)

        if download_audio and audio_root is not None:
            audio_dir = local_audio_dir(audio_root, source)
            for record in records:
                if not record.audio_file:
                    continue
                audio_dest = audio_dir / record.audio_file
                download_file(source_audio_url(source, record.audio_file), audio_dest, overwrite=overwrite)

    return downloaded_records


def parse_xml_file(xml_path: Path, source: CorpusSource) -> list[ExampleRecord]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    xml_source_label = normalize_text(root.attrib.get("source"))
    citation = normalize_text(root.attrib.get("citation"))
    copyright = normalize_text(root.attrib.get("copyright"))
    bibtex_citation = normalize_text(root.attrib.get("BibTeX_citation"))
    root_glottocode = normalize_text(root.attrib.get("glottocode"))
    root_audio_mode = normalize_text(root.attrib.get("audio"))
    language_code = normalize_text(root.attrib.get(XML_LANG_ATTR)) or "pyu"

    records: list[ExampleRecord] = []
    for sentence in root.findall("S"):
        sentence_id = normalize_text(sentence.attrib.get("id"))
        original_form = _find_child_text(sentence, "FORM", kind="original")
        standard_form = _find_child_text(sentence, "FORM", kind="standard")
        original_phonetic = _find_child_text(sentence, "PHON", kind="original")
        standard_phonetic = _find_child_text(sentence, "PHON", kind="standard")
        translation_zh = _find_child_text(sentence, "TRANSL", xml_lang="zho")
        audio_el = sentence.find("AUDIO")
        audio_file = normalize_text(audio_el.attrib.get("file") if audio_el is not None else "")
        audio_url = normalize_text(audio_el.attrib.get("url") if audio_el is not None else "")
        audio_start = safe_float(audio_el.attrib.get("start") if audio_el is not None else None)
        audio_end = safe_float(audio_el.attrib.get("end") if audio_el is not None else None)

        record_id = f"{source.source_key}:{sentence_id}"
        records.append(
            ExampleRecord(
                record_id=record_id,
                source_key=source.source_key,
                group=source.group,
                corpus_slug=source.corpus_slug,
                dialect=source.dialect,
                language_code=language_code,
                audio_mode=root_audio_mode or source.audio_mode,
                xml_source_url=source_xml_url(source),
                xml_path=str(xml_path),
                citation=citation,
                copyright=copyright,
                bibtex_citation=bibtex_citation,
                root_source_label=xml_source_label,
                root_glottocode=root_glottocode,
                sentence_id=sentence_id,
                original_form=original_form,
                standard_form=standard_form,
                original_phonetic=original_phonetic,
                standard_phonetic=standard_phonetic,
                translation_zh=translation_zh,
                audio_file=audio_file,
                audio_url=audio_url,
                audio_start=audio_start,
                audio_end=audio_end,
            )
        )

    return records


def _find_child_text(parent: ET.Element, tag: str, *, kind: str | None = None, xml_lang: str | None = None) -> str:
    for child in parent.findall(tag):
        if kind is not None and normalize_text(child.attrib.get("kindOf")) != kind:
            continue
        if xml_lang is not None and normalize_text(child.attrib.get(XML_LANG_ATTR)) != xml_lang:
            continue
        return normalize_text(child.text)
    return ""


def load_records_from_xml_root(xml_root: Path, sources: Sequence[CorpusSource] | None = None) -> list[ExampleRecord]:
    if sources is None:
        sources = supported_sources()
    records: list[ExampleRecord] = []
    for source in sources:
        xml_path = local_xml_path(xml_root, source)
        if xml_path.exists():
            records.extend(parse_xml_file(xml_path, source))
    return records


def dedupe_records(records: Sequence[ExampleRecord], *, key_fn: Callable[[ExampleRecord], str]) -> tuple[list[ExampleRecord], int]:
    seen: set[str] = set()
    kept: list[ExampleRecord] = []
    dropped = 0
    for record in records:
        key = key_fn(record)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, dropped


def translation_dedupe_key(record: ExampleRecord) -> str:
    return normalize_text(record.source_text) + "\u241f" + normalize_text(record.target_text)


def tts_dedupe_key(record: ExampleRecord) -> str:
    return normalize_text(record.audio_file) or record.record_id


def translation_row(record: ExampleRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "source_key": record.source_key,
        "group": record.group,
        "corpus_slug": record.corpus_slug,
        "dialect": record.dialect,
        "language_code": record.language_code,
        "source_text": record.source_text,
        "source_phonetic": record.source_phonetic,
        "target_text": record.target_text,
        "translation_zh": record.translation_zh,
        "audio_file": record.audio_file,
        "audio_url": record.audio_url,
        "audio_start": record.audio_start,
        "audio_end": record.audio_end,
        "duration_seconds": record.duration_seconds,
        "xml_path": record.xml_path,
        "xml_source_url": record.xml_source_url,
        "citation": record.citation,
        "copyright": record.copyright,
        "bibtex_citation": record.bibtex_citation,
    }


def tts_row(record: ExampleRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "source_key": record.source_key,
        "group": record.group,
        "corpus_slug": record.corpus_slug,
        "dialect": record.dialect,
        "language_code": record.language_code,
        "text": record.source_text,
        "phonetic": record.source_phonetic,
        "normalized_text": record.source_text,
        "audio_file": record.audio_file,
        "audio_url": record.audio_url,
        "audio_start": record.audio_start,
        "audio_end": record.audio_end,
        "duration_seconds": record.duration_seconds,
        "xml_path": record.xml_path,
        "xml_source_url": record.xml_source_url,
        "citation": record.citation,
        "copyright": record.copyright,
        "bibtex_citation": record.bibtex_citation,
    }


def _split_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_rows(
    rows: Sequence[dict[str, object]],
    *,
    key_fn: Callable[[dict[str, object]], str],
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
) -> dict[str, list[dict[str, object]]]:
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Invalid split ratios")

    train_rows: list[dict[str, object]] = []
    valid_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for row in rows:
        bucket = _split_bucket(key_fn(row))
        if bucket < train_ratio:
            train_rows.append(row)
        elif bucket < train_ratio + valid_ratio:
            valid_rows.append(row)
        else:
            test_rows.append(row)

    return {"train": train_rows, "valid": valid_rows, "test": test_rows}


def split_records(
    records: Sequence[ExampleRecord],
    *,
    key_fn: Callable[[ExampleRecord], str],
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
) -> dict[str, list[ExampleRecord]]:
    row_mapping = split_rows(
        [record.to_dict() for record in records],
        key_fn=lambda row: key_fn(_record_from_row(row)),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    split_records_map: dict[str, list[ExampleRecord]] = {name: [] for name in row_mapping}
    by_id = {record.record_id: record for record in records}
    for split_name, rows in row_mapping.items():
        for row in rows:
            split_records_map[split_name].append(by_id[str(row["record_id"])])
    return split_records_map


def _record_from_row(row: Mapping[str, object]) -> ExampleRecord:
    return ExampleRecord(
        record_id=str(row["record_id"]),
        source_key=str(row["source_key"]),
        group=str(row["group"]),
        corpus_slug=str(row["corpus_slug"]),
        dialect=str(row["dialect"]),
        language_code=str(row["language_code"]),
        audio_mode=str(row.get("audio_mode", "")),
        xml_source_url=str(row.get("xml_source_url", "")),
        xml_path=str(row.get("xml_path", "")),
        citation=str(row.get("citation", "")),
        copyright=str(row.get("copyright", "")),
        bibtex_citation=str(row.get("bibtex_citation", "")),
        root_source_label=str(row.get("root_source_label", "")),
        root_glottocode=str(row.get("root_glottocode", "")),
        sentence_id=str(row.get("sentence_id", "")),
        original_form=str(row.get("original_form", "")),
        standard_form=str(row.get("standard_form", "")),
        original_phonetic=str(row.get("original_phonetic", "")),
        standard_phonetic=str(row.get("standard_phonetic", "")),
        translation_zh=str(row.get("translation_zh", "")),
        audio_file=str(row.get("audio_file", "")),
        audio_url=str(row.get("audio_url", "")),
        audio_start=row.get("audio_start") if isinstance(row.get("audio_start"), float) else safe_float(str(row.get("audio_start")) if row.get("audio_start") is not None else None),
        audio_end=row.get("audio_end") if isinstance(row.get("audio_end"), float) else safe_float(str(row.get("audio_end")) if row.get("audio_end") is not None else None),
    )


def write_jsonl(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")


def write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    ensure_parent(path)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as fp:
            fp.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        fp.write(text)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_summary(records: Sequence[ExampleRecord]) -> dict[str, object]:
    by_group = Counter(record.group for record in records)
    by_corpus = Counter(f"{record.group}/{record.corpus_slug}" for record in records)
    by_dialect = Counter(record.dialect for record in records)
    by_audio_mode = Counter(record.audio_mode for record in records)
    missing_audio = sum(1 for record in records if not record.audio_file)
    missing_translation = sum(1 for record in records if not record.translation_zh)
    duration_sum = sum((record.duration_seconds or 0.0) for record in records)
    return {
        "total_records": len(records),
        "by_group": dict(sorted(by_group.items())),
        "by_corpus": dict(sorted(by_corpus.items())),
        "by_dialect": dict(sorted(by_dialect.items())),
        "by_audio_mode": dict(sorted(by_audio_mode.items())),
        "missing_audio": missing_audio,
        "missing_translation": missing_translation,
        "total_duration_seconds": round(duration_sum, 2),
    }


def format_summary_markdown(summary: Mapping[str, object]) -> str:
    lines = ["# Puyuma Corpus Summary", ""]
    lines.append(f"- Total records: {summary.get('total_records', 0)}")
    lines.append(f"- Missing audio: {summary.get('missing_audio', 0)}")
    lines.append(f"- Missing translation: {summary.get('missing_translation', 0)}")
    lines.append(f"- Total duration seconds: {summary.get('total_duration_seconds', 0)}")
    lines.append("")
    for label in ("by_group", "by_corpus", "by_dialect", "by_audio_mode"):
        lines.append(f"## {label}")
        entries = summary.get(label, {})
        if isinstance(entries, Mapping) and entries:
            for key, value in entries.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def materialize_ljspeech_dataset(
    rows: Sequence[Mapping[str, object]],
    *,
    audio_root: Path,
    output_dir: Path,
    text_field: str = "text",
    normalized_text_field: str = "normalized_text",
    audio_file_field: str = "audio_file",
    allow_copy: bool = True,
) -> Path:
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    ensure_parent(metadata_path)

    materialized_rows: list[list[str]] = []
    seen_audio: set[str] = set()
    for row in rows:
        audio_file = str(row.get(audio_file_field, ""))
        if not audio_file:
            continue
        if audio_file in seen_audio:
            continue
        seen_audio.add(audio_file)
        source_path = _resolve_audio_source(audio_root, row)
        dest_path = wav_dir / audio_file
        if not dest_path.exists():
            _copy_or_hardlink(source_path, dest_path, allow_copy=allow_copy)
        text = _row_text(row.get(text_field))
        normalized_text = _row_text(row.get(normalized_text_field, text))
        materialized_rows.append([audio_file, text, normalized_text])

    with metadata_path.open("w", encoding="utf-8", newline="\n") as fp:
        writer = csv.writer(fp, delimiter="|")
        for audio_file, text, normalized_text in materialized_rows:
            writer.writerow([audio_file, text, normalized_text])

    return metadata_path


def _resolve_audio_source(audio_root: Path, row: Mapping[str, object]) -> Path:
    source_key = str(row.get("source_key", ""))
    if not source_key:
        raise ValueError("Row is missing source_key")
    group, corpus_slug, _dialect = source_key.split("/", 2)
    dialect = str(row.get("dialect", ""))
    audio_repo_slug = "ILRDF_Dict_Puyuma" if group == "ILRDF_Dicts" and corpus_slug == "Puyuma" else f"ePark_{corpus_slug}"
    audio_file = str(row.get("audio_file", ""))
    if group == "ePark":
        source_path = audio_root / audio_repo_slug / "Puyuma" / f"{dialect}_Puyuma" / audio_file
    else:
        source_path = audio_root / audio_repo_slug / audio_file
    if not source_path.exists():
        raise FileNotFoundError(f"Missing audio file: {source_path}")
    return source_path


def _copy_or_hardlink(source_path: Path, dest_path: Path, *, allow_copy: bool = True) -> None:
    ensure_parent(dest_path)
    try:
        if dest_path.exists():
            return
        os.link(source_path, dest_path)
    except OSError:
        if not allow_copy:
            raise
        shutil.copy2(source_path, dest_path)


def _row_text(value: object | None) -> str:
    if value is None:
        return ""
    return normalize_text(str(value))

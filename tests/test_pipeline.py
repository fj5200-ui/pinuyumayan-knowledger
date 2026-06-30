from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from formosanbank_puyuma.pipeline import (
    ExampleRecord,
    _find_child_text,
    local_audio_dir,
    dedupe_records,
    normalize_text,
    source_audio_url,
    split_rows,
    translation_dedupe_key,
    tts_dedupe_key,
)
from formosanbank_puyuma.sources import supported_sources
from xml.etree import ElementTree as ET


class PipelineTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("  a  b \n c "), "a b c")

    def test_find_child_text_picks_kind_and_language(self) -> None:
        xml = ET.fromstring(
            """
            <S>
              <FORM kindOf="original">hello</FORM>
              <FORM kindOf="standard">world</FORM>
              <TRANSL xml:lang="zho">中文</TRANSL>
            </S>
            """
        )
        self.assertEqual(_find_child_text(xml, "FORM", kind="standard"), "world")
        self.assertEqual(_find_child_text(xml, "TRANSL", xml_lang="zho"), "中文")

    def test_dedupe_records_by_translation(self) -> None:
        record = _make_record(record_id="a", source_text="foo", target_text="bar", audio_file="1.wav")
        duplicate = _make_record(record_id="b", source_text="foo", target_text="bar", audio_file="2.wav")
        unique, dropped = dedupe_records([record, duplicate], key_fn=translation_dedupe_key)
        self.assertEqual(len(unique), 1)
        self.assertEqual(dropped, 1)

    def test_dedupe_records_by_audio(self) -> None:
        record = _make_record(record_id="a", source_text="foo", target_text="bar", audio_file="1.wav")
        duplicate = _make_record(record_id="b", source_text="baz", target_text="qux", audio_file="1.wav")
        unique, dropped = dedupe_records([record, duplicate], key_fn=tts_dedupe_key)
        self.assertEqual(len(unique), 1)
        self.assertEqual(dropped, 1)

    def test_split_rows_is_deterministic(self) -> None:
        rows = [{"record_id": f"id-{index}", "source_text": f"s{index}", "target_text": f"t{index}"} for index in range(50)]
        split_a = split_rows(rows, key_fn=lambda row: str(row["record_id"]), train_ratio=0.8, valid_ratio=0.1)
        split_b = split_rows(rows, key_fn=lambda row: str(row["record_id"]), train_ratio=0.8, valid_ratio=0.1)
        self.assertEqual(split_a, split_b)
        self.assertEqual(sum(len(part) for part in split_a.values()), len(rows))

    def test_epark_audio_path_is_nested(self) -> None:
        source = supported_sources()[0]
        audio_url = source_audio_url(source, "demo.wav")
        self.assertIn("/resolve/main/Puyuma/Nanwang_Puyuma/demo.wav", audio_url)

        audio_dir = local_audio_dir(Path("data/raw/audio"), source)
        self.assertEqual(
            audio_dir.as_posix(),
            "data/raw/audio/ePark_hui_ben_ping_tai_picture_book_platform/Puyuma/Nanwang_Puyuma",
        )


def _make_record(record_id: str, source_text: str, target_text: str, audio_file: str) -> ExampleRecord:
    return ExampleRecord(
        record_id=record_id,
        source_key="ePark/demo/Nanwang",
        group="ePark",
        corpus_slug="demo",
        dialect="Nanwang",
        language_code="pyu",
        audio_mode="diarized",
        xml_source_url="https://example.com/source.xml",
        xml_path="C:/tmp/source.xml",
        citation="citation",
        copyright="copyright",
        bibtex_citation="bibtex",
        root_source_label="source",
        root_glottocode="",
        sentence_id=record_id,
        original_form=source_text,
        standard_form=source_text,
        original_phonetic=source_text,
        standard_phonetic=source_text,
        translation_zh=target_text,
        audio_file=audio_file,
        audio_url="https://example.com/audio.wav",
        audio_start=0.0,
        audio_end=1.0,
    )


if __name__ == "__main__":
    unittest.main()

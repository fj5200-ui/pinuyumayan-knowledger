from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

GITHUB_ORG = "FormosanBank"
GITHUB_REPO = "FormosanBank"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_ORG}/{GITHUB_REPO}/main/Corpora"
HF_DATASET_BASE = "https://huggingface.co/datasets"
XML_LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"


@dataclass(frozen=True, slots=True)
class CorpusSource:
    group: str
    corpus_slug: str
    dialect: str
    xml_relative_path: str
    audio_repo: str
    audio_mode: str

    @property
    def source_key(self) -> str:
        return f"{self.group}/{self.corpus_slug}/{self.dialect}"

    @property
    def xml_url(self) -> str:
        return f"{GITHUB_RAW_BASE}/{self.xml_relative_path}"

    @property
    def audio_repo_slug(self) -> str:
        return self.audio_repo.split("/", 1)[1]

    @property
    def audio_base_url(self) -> str:
        return f"{HF_DATASET_BASE}/{self.audio_repo}"

    @property
    def local_xml_relative_dir(self) -> str:
        parts = self.xml_relative_path.split("/")
        return "/".join(parts[:-1])


E_PARK_CORPORA = {
    "hui_ben_ping_tai_picture_book_platform": ("Nanwang",),
    "jiu_jie_jiao_cai_nine_level_materials": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "ju_xing_pian_gao_zhong_sentence_patterns_senior_high": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "ju_xing_pian_guo_zhong_sentence_patterns_junior_high": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "qing_jing_zu_yu_contextual_indigenous_language": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "sheng_huo_hui_hua_pian_daily_conversation": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "tu_hua_gu_shi_pian_picture_story": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "wen_hua_pian_cultural_section": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "xue_xi_ci_biao_learning_vocabulary": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "yue_du_shu_xie_pian_reading_writing": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
    "zu_yu_duan_wen_indigenous_language_essays": ("Jianhe", "Nanwang", "Xiqun", "Zhiben"),
}


def iter_supported_sources() -> Iterator[CorpusSource]:
    for corpus_slug, dialects in E_PARK_CORPORA.items():
        for dialect in dialects:
            yield CorpusSource(
                group="ePark",
                corpus_slug=corpus_slug,
                dialect=dialect,
                xml_relative_path=f"ePark/XML/{corpus_slug}/Puyuma/{dialect}_Puyuma.xml",
                audio_repo=f"FormosanBank/ePark_{corpus_slug}",
                audio_mode="diarized",
            )

    yield CorpusSource(
        group="ILRDF_Dicts",
        corpus_slug="Puyuma",
        dialect="Nanwang",
        xml_relative_path="ILRDF_Dicts/XML/Puyuma/Puyuma.xml",
        audio_repo="FormosanBank/ILRDF_Dict_Puyuma",
        audio_mode="segmented",
    )


def supported_sources() -> list[CorpusSource]:
    return list(iter_supported_sources())

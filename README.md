# FormosanBank Puyuma TTS

這個專案把 FormosanBank 裡的卑南族（Puyuma）語料與音檔整理成可重跑的資料管線，目標是同時支援：

- 翻譯資料集抽取與切分
- TTS 訓練資料準備
- 翻譯模型微調
- TTS 模型微調骨架

## 資料來源

專案預設從 FormosanBank 的公開來源重建資料：

- GitHub XML 原始檔: `FormosanBank/FormosanBank`
- Hugging Face 音檔資料集: `FormosanBank/ILRDF_Dict_Puyuma` 與各 `FormosanBank/ePark_*` dataset

注意：

- FormosanBank 資料含有非商業使用限制。
- 這個 repo 只放程式與 manifest，不把原始語料與音檔直接提交進 Git。

## 主要輸出

- `data/raw/xml/` - 下載下來的 XML
- `data/raw/audio/` - 下載下來的音檔，ePark 會存成 `ePark_<corpus>/Puyuma/<Dialect>_Puyuma/`
- `data/processed/puyuma_examples.jsonl` - 原始例句總表
- `data/processed/translation_*.jsonl` - 翻譯訓練切分
- `data/processed/tts_*.jsonl` - TTS 訓練切分
- `data/tts_ljspeech/` - 可給 Coqui TTS 使用的 LJSpeech 風格資料夾

## 快速開始

先確認你本機 Python 3.12 可用，然後執行：

```powershell
python -m formosanbank_puyuma.cli list-sources
python -m formosanbank_puyuma.cli prepare --data-dir data --download-audio
python -m formosanbank_puyuma.cli stats --input data/processed/puyuma_examples.jsonl
python -m formosanbank_puyuma.cli split --input data/processed/puyuma_examples.jsonl --output-dir data/processed
```

如果你要準備 TTS 訓練資料：

```powershell
python -m formosanbank_puyuma.cli prepare-tts \
  --input data/processed/tts_train.jsonl \
  --audio-root data/raw/audio \
  --output-dir data/tts_ljspeech
```

## 翻譯訓練

預設翻譯模型使用 ByT5 類型的 byte-level seq2seq 架構，適合低資源語言：

```powershell
python -m formosanbank_puyuma.train_translation \
  --train-file data/processed/translation_train.jsonl \
  --validation-file data/processed/translation_valid.jsonl \
  --model-name google/byt5-small \
  --output-dir runs/translation
```

## TTS 訓練

這個專案會先把資料整理成 LJSpeech 風格，並輸出 Coqui TTS 可用的 config baseline。你可以依實際模型版本調整 `text_cleaner`、`sample_rate` 和 `restore_path`。

```powershell
python -m formosanbank_puyuma.train_tts \
  --manifest data/processed/tts_train.jsonl \
  --audio-root data/raw/audio \
  --output-dir runs/tts \
  --run-name puyuma-tts
```

如果你要真的開訓練，建議先安裝：

```powershell
python -m pip install .[translate,tts]
```

`tts` extra 會安裝 Coqui TTS 的維護版 `coqui-tts`，較適合 Windows / Python 3.12。

## Repo 結構

```text
src/formosanbank_puyuma/
  cli.py
  pipeline.py
  sources.py
  train_translation.py
  train_tts.py
tests/
```

## 備註

- 這個 repo 預設是研究用途與資料準備工具。
- 如果你要正式訓練，請先確認你要使用的 Coqui TTS / Transformers 版本與授權條款。

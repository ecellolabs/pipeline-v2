# Dataset Reference: MMLongBench-Doc, MP-DocVQA, SlideVQA

Compiled 2026-09-18 from the published papers, official repos/portals, HF dataset cards, and follow-up papers. Each section covers: (1) split distribution, (2) data description, (3) published evaluation metrics, (4) original format and what `pipeline-v2` takes from it, (5) leaderboard architectures, (6) Qwen performance.

Loader code referenced: `src/pipeline_v2/datasets/{mmlongbench_doc,mpdocvqa,slidevqa}.py`. All three loaders map into `atria_core`'s `MultiPageDocumentInstance` with one `MultiPageQuestionAnsweringAnnotation` per document, whose `MultiPageQAPair` schema is:
`id, question_text, answer_text, evidence_pages (0-indexed), arithmetic_expression, alternative_answers, evidence_sources, answer_format`.

---

## 1. MMLongBench-Doc

Paper: Ma et al., *MMLongBench-Doc: Benchmarking Long-context Document Understanding with Visualizations*, NeurIPS 2024 D&B (Spotlight). arXiv 2407.01523. GitHub `mayubo2333/MMLongBench-Doc`. HF `yubo2333/MMLongBench-Doc`.

### 1.1 Distribution

**Single evaluation set. No train/dev/test split.** The HF repo exposes one `train` split purely for `load_dataset` convenience; the pipeline loader accordingly registers only `DatasetSplitType.train`.

Dataset size has drifted across versions (sources mix them):

| Version | Docs | Questions | Unanswerable | Cross-page | Single-page |
|---|---|---|---|---|---|
| arXiv v1 (2024-07-01) | 130 | 1,062 | 242 (22.8%) | 353 (33.3%) | – |
| arXiv v3 / NeurIPS camera-ready (2024-11-12) | 135 | 1,082 | 223 (20.6%) | 365 (33.7%) | 494 (45.7%) |
| GitHub `data/samples.json` (current) | 135 | 1,082 | 223 | 360 | – |
| **HF parquet after 09/2025 Q&A refinement (what the pipeline loads)** | **135** | **1,091** | **244 (22.4%)** | **360 (33.0%)** | **485 (44.5%)** |

Sources: arXiv abs v1/v3, NeurIPS proceedings page, direct counts of `samples.json` and the HF rows via datasets-server.

### 1.2 Data description

- **Document types** (7, counted from `samples.json`; questions column = GitHub 1,082 / HF 1,091):

| doc_type | Docs | % docs | Questions |
|---|---|---|---|
| Research report / Introduction | 34 | 25.2% | 292 / 293 |
| Academic paper | 26 | 19.3% | 199 / 204 |
| Guidebook | 22 | 16.3% | 155 / 156 |
| Tutorial/Workshop | 17 | 12.6% | 138 / 139 |
| Brochure | 15 | 11.1% | 100 / 101 |
| Financial report | 11 | 8.1% | 117 / 117 |
| Administration/Industry file | 10 | 7.4% | 81 / 81 |

- **Sources**: 76 docs re-used from DUDE, SlideVQA, ChartQA, FinanceBench (184 derived questions); 59 newly collected from arXiv, ManualsLib and Google Search. Docs with <15 pages or licence restrictions filtered out. 898 questions (83%) newly annotated by ten PhD-level annotators.
- **Scanned vs digital**: not stated explicitly by the paper. Sources (arXiv papers, 10-K filings, Pew reports, manuals, slide decks) are overwhelmingly born-digital PDFs; the DUDE-derived subset may contain scans. Official evaluation renders pages to PNG at 144 DPI; the OCR baseline uses Tesseract.
- **Length** (v3 Table 2): avg/median pages 47.5 / 28; avg/median tokens 21,214 / 12,179; max ~120 pages. Total pages ≈ 6.4k (derived).
- **Question characteristics** (v3, 1,082 Qs): evidence sources multi-label: Pure-text 305 (35.5%), Layout 119 (13.9%), Table 218 (25.4%), Chart 178 (20.7%), Image/Figure 304 (35.4%). Answer formats (answerable): Str 250 (29.1%), Int 299 (34.8%), Float 159 (18.5%), List 151 (17.6%); unanswerable → `None`. HF-current: Int 290, Str 250, None 244, Float 160, List 147. Avg question 16.4 words; avg answer 2.8 words.
- **Release**: HF repo created 2024-06-12; arXiv v1 2024-07-01, v3 2024-11-12; Q&A refinement + official leaderboard launched 09/2025.
- **Licence**: data CC BY-NC 4.0 (research use only), code Apache-2.0.
- **Size**: HF repo ≈ 662 MB total; `documents/` holds 194 PDFs ≈ 0.51 GB (only 135 are referenced by `doc_id`); question parquet 82 KB.

### 1.3 Published evaluation metrics

Three-stage protocol (follows MathVista), official code under `eval/`:

1. **Free-form response generation**, temperature 0, `max_new_tokens` 1024, model asked to be concise.
2. **GPT-4o answer extraction** (`eval/prompt_for_answer_extraction.md`): given question + model analysis, emit `Extracted answer: [...]` and `Answer format: [Integer|Float|String|List]`; emit "Not answerable" when the analysis says the doc cannot answer, "Fail to answer" when the model only says it can't read the images. Four few-shot examples.
3. **Rule-based scoring** (`eval/eval_score.py`), keyed on `answer_format`:
   - `Int`: `int(gt) == int(float(pred))`.
   - `Float`: strip `$ % ( )`, then `isclose(rel_tol=0.01)` against gt, gt/100, gt×100, or rounding to ≥2 decimals.
   - `Str` / `None`: lowercase, strip quotes/`$`/`%`/parentheticals; if GT looks like a URL, filename, "page …", phone, time, ISO date or email → exact match; else **ANLS** = 1 − Levenshtein/maxlen, zeroed when ≤ 0.5. GT "Not answerable" scores 1 if predicted, 0 for "Fail to answer".
   - `List`: length must match; elements cleaned and sorted; numeric → joined exact match; else `min` of element-wise ANLS.
   - **Accuracy** = mean score over all questions (fractional ANLS allowed, so "generalized accuracy").
   - **F1**: recall = Σscore(answerable)/#answerable; precision = Σscore(answerable)/#(pred ≠ "Not answerable"); harmonic mean. Human audit found 4–6 extraction disagreements per 100 responses.

### 1.4 Original format and what the pipeline takes

**Original HF/GitHub row** (all string-typed): `doc_id` (PDF filename), `doc_type`, `question`, `answer`, `evidence_pages` (stringified list, 1-indexed, `"[]"` when unanswerable), `evidence_sources` (stringified list from {Pure-text (Plain-text), Generalized-text (Layout), Table, Chart, Figure}), `answer_format` ∈ {Int, Float, Str, List, None}. PDFs live under `documents/`.

**Taken by `mmlongbench_doc.py`:**

| Original field | Destination | Transform |
|---|---|---|
| `doc_id` | `sample_id`, `metadata.doc_id`; groups rows into one document | – |
| PDF `documents/<doc_id>` | `MultiPageDocumentInstance.from_pdf` (pages rendered by atria_core) | downloaded per-doc from HF `resolve/main/documents/` into `<data_dir>/pdfs/` |
| `doc_type` | `metadata.doc_type` | taken from the first row of the doc |
| `question` / `answer` | `question_text` / `answer_text` | – |
| `evidence_pages` | `evidence_pages` | `json.loads`, then **1-indexed → 0-indexed** |
| `evidence_sources` | `evidence_sources` | `ast.literal_eval` |
| `answer_format` | `answer_format` | – |
| (none) | `id` | enumeration index within the document |

Dropped: nothing substantive. `alternative_answers` and `arithmetic_expression` stay empty. Note the pipeline loads whatever is on HF `main` today (1,091 rows), not the frozen 1,082-question NeurIPS set.

### 1.5 Leaderboard architectures

**Paper Table 3 (v3, Acc / F1).** LVLMs receive per-page 144-DPI PNGs; open-source LVLMs get all pages concatenated into 1 or 5 images; Claude-3 Opus capped at 20 images; GPT-4o/4V/Gemini get every page as an image. LLM pipelines get Tesseract OCR text.

| Category | Model | Acc | F1 |
|---|---|---|---|
| OCR+LLM (open) | Mixtral 8x22B | 26.9 | 24.7 |
| OCR+LLM (proprietary) | GPT-4o | 30.1 | 30.5 |
| | Gemini-1.5-Pro | 31.2 | 24.8 |
| | Claude-3 Opus | 26.9 | 24.5 |
| LVLM 7–14B | InternLM-XC2-4KHD 8B | 10.3 | 9.8 |
| | MiniCPM-Llama3-V2.5 8B | 8.5 | 8.6 |
| | Qwen-VL-Chat 9.6B | 6.1 | 5.4 |
| LVLM >14B | InternVL-Chat-v1.5 26B | 14.6 | 13.0 |
| LVLM (proprietary) | GPT-4o | **42.8** | **44.9** |
| | GPT-4V | 32.4 | 31.2 |
| | Gemini-1.5-Pro | 28.2 | 20.6 |
| | Claude-3 Opus | 17.4 | 18.1 |
| Human | experts | 65.8 | 66.0 |

Key finding: 12 of 14 open LVLMs scored *below* their OCR+LLM counterparts.

**Official leaderboard** (HF Space `OpenIXCLab/mmlongbench-doc`, Acc only, as of 2026-09): Claude 4.5 Opus 61.9 · Qwen3.5-397B-A17B 61.5 · Gemini-3 Pro 60.5 · OriOn-Qwen-SR1 32B 58.3 · Nemotron 3 Nano Omni 30B-A3B 57.6 · Qwen3-VL-235B-A22B-Instruct 57.0 · Qwen3-VL-235B-A22B-Thinking 56.2 · TeleMM-2.0 56.1 · GLM-4.6V 54.9 · GPT-4.1 49.7 · GPT-4o (2024-11-20) 46.3 · GLM-4.5V 44.7 · Kimi-VL-Thinking-2506 42.1 · Qwen2.5-VL-72B 35.2 · MiniMax-VL-01 32.5 · Aria 28.3 · Qwen2.5-VL-7B 25.1.

Architecture families seen: (a) end-to-end LVLMs with all pages as images vs concatenated-image input; (b) OCR/PDF-text + LLM; (c) retrieval-augmented (ColPali page retrieval + VLM, e.g. M3DocRAG) and agentic/tool frameworks (MDocAgent, SimpleDoc, DocLens 63–68 with Gemini-2.5-Pro, surpassing the human 65.8) — these often use an LLM-judge protocol rather than the official one, so numbers are not strictly comparable.

### 1.6 Qwen performance

| Model | Acc | F1 | Protocol / source |
|---|---|---|---|
| Qwen-VL-Chat 9.6B | 6.1 | 5.4 | Official, paper Table 3 |
| Qwen-Plus (OCR+LLM) | 18.9 | 13.4 | Official, paper v3 |
| Qwen2-VL-7B | 21.3 | 22.7 | Aria report Table 3 (arXiv 2410.05993) |
| ColPali + Qwen2-VL-7B top-1 / top-4 | 18.8 / 21.0 | 20.1 / 22.6 | M3DocRAG Table 2 (arXiv 2411.04952), official scoring |
| Qwen2-VL-7B full-doc | 16.5 | – | MDocAgent, GPT-4o binary judge (not official) |
| Qwen2.5-VL-7B | **25.1** | – | Official leaderboard |
| Qwen2.5-VL-7B | 29.6 | – | Kimi-VL report Table 3 (arXiv 2504.07491) |
| Qwen2.5-VL-32B no-RAG / oracle pages | 22.2 / 67.9 | – | SimpleDoc Table 1, GPT-4.1 judge |
| Qwen2.5-VL-72B | **35.2** | – | Official leaderboard |
| Qwen3-VL-2B Thinking / Instruct | 33.8 / 31.6 | – | Qwen3-VL tech report Table 4 (arXiv 2511.21631) |
| Qwen3-VL-4B Thinking / Instruct | 44.4 / 43.5 | – | same |
| Qwen3-VL-8B Thinking / Instruct | 48.0 / 47.9 | – | same |
| Qwen3-VL-30B-A3B Thinking / Instruct | 47.4 / 47.1 | – | same, Table 3 |
| Qwen3-VL-32B Thinking / Instruct | 54.6 / 55.4 | – | same, Table 3 |
| Qwen3-VL-235B-A22B Thinking / Instruct | 56.2 / 57.0 | – | same, Table 2; matches official leaderboard |
| Qwen3.5-397B-A17B | 61.5 | – | Official leaderboard |

Not found: Qwen2-VL-72B, Qwen2.5-VL-3B. The Qwen2.5-VL technical report and HF model cards do not list MMLongBench-Doc. F1 is generally not published for post-2024 models.

---

## 2. MP-DocVQA

Paper: Tito, Karatzas, Valveny, *Hierarchical multimodal transformers for Multi-Page DocVQA*, Pattern Recognition 2023. arXiv 2212.05935. GitHub `rubenpt91/MP-DocVQA-Framework`. RRC portal ch=17, Task 4.

### 2.1 Distribution

Paper totals (v2, App. A Table 4): **46,176 questions over 47,952 page images from 5,928 documents** after the 20-page cap (from 50,000 SP-DocVQA questions; 3,824 docs / 39,688 questions are multi-page only). The paper gives no per-split table; it says the split keeps SP-DocVQA's distribution and no document is shared between train and val/test.

Per-split counts from the author's HF card (`rubentito/mp-docvqa`), corroborated for val/test by `lmms-lab/MP-DocVQA` row counts and DocR1 Table 7:

| Split | Questions | % | Documents | Pages |
|---|---|---|---|---|
| Train | 36,230 | 78.0% | 5,131 | 37,269 |
| Validation | 5,187 | 11.2% | 927 | 6,510 |
| Test | 5,019 | 10.8% | 959 | 6,223 |
| Sum | 46,436 | | 7,017 | 50,002 |

Caveat: the card's sums exceed the paper totals (46,436 vs 46,176 questions; docs/pages overlap because "some documents appear in both validation and test sets but never in training").

Other stats (Table 1): avg 8.27 pages per question, 85.95% of questions on multi-page docs, avg question 9.90 words, avg answer 2.20 words, avg 2,027 OCR tokens per document (range 1–42,313).

### 2.2 Data description

- **Source**: extends SingleDocVQA (DocVQA Task 1). Industry documents from the UCSF Industry Documents Library; authors appended the previous/posterior pages of each original page from UCSF-IDL, capped at 20 pages. Questions and answers are reused verbatim from Task 1.
- **Document nature**: **scanned** industry documents (tobacco/industry archives): printed, typewritten and handwritten text; forms, tables, lists, diagrams, pictures.
- **Pages per doc**: 1–20 (capped), avg 8.27.
- **Image format**: JPG, one per page, `<doc_id>_p<k>.jpg`.
- **OCR**: Amazon Textract for all 47,952 images, shipped both as a raw OCR archive and pre-baked into IMDB `.npy` files.
- **Download sizes** (CVC file server): `images.tar.gz` ≈ 21.2 GiB; `ocr.tar.gz` ≈ 2.84 GiB; `mpdocvqa_imdbs.zip` ≈ 1.67 GiB; `qas.zip` ≈ 1.7 MiB. lmms-lab HF mirror (val+test only): 8.6 GB parquet / 25.5 GB decoded.
- **Release**: arXiv v1 2022-12-07; RRC portal "MP-DocVQA Dataset released" 2023-02-19; Pattern Recognition 2023.
- **Licence/terms**: portal registration required; no explicit licence text on public pages (author's HF card metadata says MIT, but that repo holds no data).

### 2.3 Published evaluation metrics

- **ANLS** (Average Normalized Levenshtein Similarity, from ST-VQA): ANLS = (1/N) Σ_i max_j s(a_ij, o_i), with s(a,o) = 1 − NL(a,o) if NL(a,o) < 0.5 else 0; NL = Levenshtein distance / length of the longer string; max over all ground-truth answers. Case-insensitive, space-sensitive. Framework `metrics.py` implements exactly this (`anls_threshold = 0.5`).
- **Answer Page Prediction Accuracy (APPA)**: fraction of questions whose predicted page index equals `answer_page_idx` (index into `page_ids`). Optional in submissions; entries that don't predict a page show 50.79% (page 0 default).
- Test-set scoring only via the RRC server.

### 2.4 Original format and what the pipeline takes

**Original distribution**: (a) `qas.zip` question JSONs (`questionId, question, answers, image, doc_id, page_ids[1..20], answer_page_idx, data_split`); (b) `images.tar.gz` JPGs; (c) `ocr.tar.gz`; (d) `mpdocvqa_imdbs.zip` → `imdb_{train,val,test}.npy`: `data[0]` header, `data[1:]` per-question dicts with `question_id, question, answers, answer_page_idx, imdb_doc_pages, image_id, image_name[], ocr_tokens[page][tok], ocr_normalized_boxes[page][tok][x0,y0,x1,y1]` (0–1).

**Taken by `mpdocvqa.py`** — uses only (b) images and (d) IMDBs (auto-downloads the IMDBs; images require manual RRC download):

| Original field | Destination | Transform |
|---|---|---|
| `image_id` | `sample_id`; groups records into one document | – |
| `image_name[]` + `images/<name>.jpg` | one `SinglePageDocumentInstance.from_image` per page, `sample_id = <image_id>#<page>` | – |
| `ocr_tokens` / `ocr_normalized_boxes` | per-page `DocumentContent` via `ElementArray.from_words` (XYXY, normalized) | taken from the first record of the doc (identical across records) |
| `question_id` / `question` | `id` / `question_text` | – |
| `answers[]` | `answer_text = answers[0]`, `alternative_answers = answers[1:]` | de-duplicated, order-preserving |
| `answer_page_idx` | `evidence_pages = [idx]` | already 0-indexed; empty on test |
| test split | answers/evidence left empty | `has_answer=False` records |

Dropped: `imdb_doc_pages`, header dict, raw OCR archive, `qas.zip` JSONs, `data_split`. All three splits are exposed (`train`, `validation`→`val`, `test`).

### 2.5 Leaderboard architectures

**Paper Table 2 (test)**: Hi-VT5 0.6201 ANLS / 79.23 APPA (multipage); BigBird concat 0.4929; Longformer concat 0.5287; LayoutLMv3 concat 0.4538; T5 concat 0.5050; oracle-page upper bounds 0.59–0.68.

**RRC Task 4 leaderboard** (test, fetched 2026-09-18):

| Date | Method | ANLS | Page Acc |
|---|---|---|---|
| 2025-11 | RealDoc-PageTreeIndex (Zoloz) | **0.8823** | 82.49 |
| 2025-06 | INF-InfoExtractor | 0.8801 | 51.07 |
| 2025-03 | AVIR-Qwen2.5-VL-7B | 0.8763 | 81.63 |
| 2025-02 | Qwen2.5-VL-7B-AWQ-lite | 0.8752 | 50.79 |
| 2024-12 | qwen2vl-2b ensemble | 0.8501 | **85.95** |
| 2025-03 | AVIR-Qwen-2.5-VL-3B | 0.8458 | 50.79 |
| 2025-03 | Qwen2.5-VL-3B-Instruct-AWQ | 0.8405 | 50.79 |
| 2024-08 | Snowflake Arctic-TILT 0.8B | 0.8122 | 50.79 |
| 2024-01 | GRAM | 0.8032 | 19.98 |
| 2024-02 | ScreenAI 5B | 0.7711 | 77.88 |
| 2025-09 | MP-DIVE-Doc 2.6B | 0.7072 | 76.25 |
| 2025-01 | mPLUG-DocOwl2 | 0.6932 | 50.79 |
| 2023-10 | OCR-free retrieval baseline | 0.6199 | 81.55 |
| 2023-03 | Hi-VT5 | 0.6184 | 79.64 |
| 2023-02 | Longformer concat baseline | 0.5287 | 71.17 |

Architectures:
- **RealDoc-PageTreeIndex**: DeepSeek-OCR-style compact visual text encoding + vectorless reasoning-based hierarchical tree index for page retrieval, VLM answerer.
- **INF-InfoExtractor**: Infinity-Parser-7B page→markdown, inf-retriever-v1-1.5b page retrieval, Qwen2.5-VL-7B answers on selected page images.
- **AVIR** (arXiv 2601.11976): ~0.1B page-retrieval model (SelfAttnScoring) + adaptive page selector feeding a frozen AWQ Qwen2.5-VL; avg 2.9 pages/question, no task fine-tuning.
- **Arctic-TILT 0.8B**: sub-billion TILT encoder-decoder for long business docs.
- **GRAM** (CVPR 2024): DocFormerv2 backbone with global document-level layers and C-Former compression.
- **Hi-VT5**: T5-base + DiT visual features; each page encoded into 10 `[PAGE]` tokens, decoder attends across all pages (up to 20,480 tokens), extra head predicts answer page.
- **M3DocRAG** (ColPali + Qwen2-VL-7B): 0.8444 ANLS / 81.05 page R@1 on the test server (not on the public board).
- **SV-RAG**: single MLLM with two LoRA adapters (retriever + QA), InternVL2-4B, ~0.70 ANLS.

### 2.6 Qwen performance (ANLS unless noted)

| Source | Model / setting | ANLS | Split |
|---|---|---|---|
| RRC leaderboard | Qwen2.5-VL-7B-AWQ-lite | 0.8752 | test |
| RRC leaderboard | AVIR-Qwen2.5-VL-7B | 0.8763 (page 81.63) | test |
| RRC leaderboard | AVIR-Qwen-2.5-VL-3B | 0.8458 | test |
| RRC leaderboard | Qwen2.5-VL-3B-Instruct-AWQ | 0.8405 | test |
| RRC leaderboard | qwen2vl-2b ensemble | 0.8501 (page 85.95) | test |
| M3DocRAG Tab. 3 | ColPali + Qwen2-VL-7B | 0.8444 | test |
| DocVLM Tab. 2/6 (arXiv 2412.08746) | Qwen2-VL-7B zero-shot, 256 / 1024 tok per page | 73.0 / 85.2 | test |
| DocR1 Tab. 4 (arXiv 2508.07313) | Qwen2.5-VL-7B / 32B; DocR1-7B | 87.39 / 84.79; 87.45 | val |
| URaG Tab. 2 (arXiv 2511.10552) | Qwen2-VL-7B / Qwen2.5-VL-3B / 7B; URaG-3B / 7B | 82.1 / 84.4 / 87.2; 86.0 / 88.2 | unstated |
| Doc-V* Tab. 1 (arXiv 2604.13731) | Qwen2.5-VL-7B; Doc-V* SFT / GRPO; GPT-4o | 75.2; 81.3 / 86.2; 67.4 | unstated |
| RAG-DocVQA Tab. 2 (arXiv 2508.18984) | Qwen2.5-VL-7B concat; RAG-Qwen2.5-VL | 51.2; 73.7 | val |

The spread for the same Qwen2.5-VL-7B (51.2 → 87.4) comes from `max_pixels`/resolution, page packing, and whether a retriever is used. No Qwen3-VL number on MP-DocVQA was found.

Reference only (single-page DocVQA test, not MP-DocVQA): Qwen2-VL 2B/7B/72B 90.1/94.5/96.1; Qwen2.5-VL 3B/7B/72B 93.9/95.7/96.4; Qwen3-VL 8B-Instruct 96.1, 235B-A22B-Instruct 97.1.

---

## 3. SlideVQA

Paper: Tanaka et al., *SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images*, AAAI 2023. arXiv 2301.04883. GitHub `nttmdlab-nlp/SlideVQA`. HF `NTT-hil-insight/SlideVQA` (gated).

### 3.1 Distribution

| Split | Questions | % | HF split / shards |
|---|---|---|---|
| train | 10,617 | 73.3% | `train`, 53 parquet shards |
| dev | 1,652 | 11.4% | `val`, 9 shards |
| test | 2,215 | 15.3% | `test`, 12 shards |
| **Total** | **14,484** | | |

Each deck appears in exactly one split. Per-split deck/image counts are not published (2,619 decks / 52,480 images overall ≈ 20 per deck); they can be computed from the parquet `deck_name` column. The pipeline's `_HF_SPLIT_NAMES` maps `validation` → HF `val` and `_BBOX_SPLIT_NAMES` maps it → GitHub `dev.jsonl`.

### 3.2 Data description

- **Source**: SlideShare. 25,327 decks with >20 slides across **39 topics** were pulled, truncated to the first 20 slides, then crowd-filtered for English, understandability, and presence of graphs/tables/figures/numeric data.
- **Final size**: 2,619 decks, 52,480 slide images, 14,484 QA pairs, 890,945 bounding boxes, ~1.7k arithmetic-expression annotations, 20 evidence candidates per question.
- **Image type**: **born-digital** slide renders served by SlideShare's CDN. Pixel resolution is not stated anywhere official.
- **Reasoning types**: Single-hop, Multi-hop, Single-hop & Numerical, Multi-hop & Numerical; 49.3% of questions need multi-hop or numerical reasoning; numerical ops = Arithmetic, Count, Comparison; 25.5% of numerical questions need arithmetic. 2,018 multi-hop questions were created by editing single-hop ones.
- **Answer types**: Single-span, Multi-span, Non-span; multi-span + non-span = 32.4% of test answers.
- **Bounding-box classes** (9, SPaSe-based): Title, Page-text, Obj-text, Caption, Other-text, Diagram, Table, Image, Figure.
- **OCR** (for baselines): Google Cloud Vision, avg 1,489 OCR tokens per deck.
- **Size**: HF `download_size` 10.9 GB parquet, `dataset_size` 36.7 GB decoded.
- **Release**: arXiv v1 2023-01-12; AAAI-23 proceedings 2023-06-26; HF parquet release 2025-03-26 (because many SlideShare URLs now 404).
- **Licence**: **not CC BY-NC-SA** (the loader's `_LICENSE` constant is inaccurate). Both GitHub `LICENSE` and the gated HF repo use NTT's *Software Evaluation License Agreement*: royalty-free, internal testing/evaluation only, no redistribution/modification.

### 3.3 Published evaluation metrics

From the paper and official `evaluate.py`:

- **QA task**: Exact Match (EM) and token-level F1 (HotpotQA style). `normalize_answer()` lowercases, strips punctuation/articles, maps number words to digits, strips units after "how many"/"which".
- **Evidence selection task**: EM and F1 over predicted vs gold `evidence_pages` sets (precision = hits/|pred|, recall = hits/|gold|). Recall@3 = 96.0% is quoted only for the H-LayoutLMv2 selector used by pipeline baselines.
- **Main (joint) task**: Joint EM and Joint F1 — joint precision/recall = QA × evidence-selection precision/recall; JEM requires both to be exact.
- **Arithmetic mode**: model may emit `Expression: …`; a calculator evaluates it and the resulting number is scored with EM/F1. The script scores final answers only.

### 3.4 Original format and what the pipeline takes

**Original**:
- HF parquet, one row per question with the deck's images duplicated per row: `deck_name, deck_url, page_1 … page_20 (image), qa_id, question, answer, arithmetic_expression ("None" when absent), evidence_pages (1-indexed int list)`.
- GitHub QA jsonl adds `image_urls, reasoning_type, answer_type` (the latter two are **not** in the parquet).
- GitHub bbox jsonl (`annotations/bbox/{train,dev,test}.jsonl`): `deck_name, deck_url, image_urls, category, bboxes = [[page_no, [{bbox_id, class, bbox:[x, y, w, h] px}]]]`.
- Images originally via `download_slides_slideshare.py` from `image_urls`.

**Taken by `slidevqa.py`:**

| Original field | Destination | Transform |
|---|---|---|
| `deck_name` | `sample_id`, `metadata.deck_name`; groups rows into one deck | – |
| `page_1..page_20` | per-page `SinglePageDocumentInstance.from_image`, `sample_id = <deck>#<page>` | decoded once per deck (first row), saved as `images/<split>/<deck>/<n>.png` |
| bbox jsonl (GitHub) | per-page `ObjectDetectionAnnotation` (XYWH, unnormalized; `label_name = class`) | downloaded from raw GitHub, indexed by byte offset per deck; `page_no` 1→0-indexed |
| `qa_id` / `question` / `answer` | `id` / `question_text` / `answer_text` | – |
| `arithmetic_expression` | `arithmetic_expression` | `"None"` → `None` |
| `evidence_pages` | `evidence_pages` | **1-indexed → 0-indexed** |

Dropped: `deck_url`, `image_urls`, `category`, `reasoning_type`, `answer_type` (only in GitHub jsonl), `bbox_id`. `evidence_sources`, `answer_format`, `alternative_answers` stay empty. With `--max-samples N` only the first ⌈N/25⌉ parquet shards are fetched (25 rows per shard assumed).

### 3.5 Leaderboard architectures

**Paper Table 2 (test)**:

| Task | Model | EM | F1 |
|---|---|---|---|
| QA | Q-only | 10.7 | 13.5 |
| QA | T5 (OCR text) | 29.3 | 37.9 |
| QA | LayoutT5 | 31.7 | 39.9 |
| QA | LayoutLMv2 (extractive) | 21.4 | 29.3 |
| QA | FiD | 30.4 | 38.9 |
| QA | **M3D** | **33.5** | **41.7** |
| QA | Human | 89.8 | 93.0 |
| Evidence sel. | H-LayoutLMv2 | 69.8 | 85.6 |
| Evidence sel. | M3D | 75.0 | 83.8 |
| Joint | M3D (JEM/JF1) | 28.0 | 37.3 |
| Joint | Human | 88.6 | 91.9 |

M3D = Fusion-in-Decoder (T5-base) over Faster-RCNN regions + OCR tokens + layout/visual features, multi-task evidence selection + QA, arithmetic-expression generation (dev QA 41.3 / 47.1).

**Later results (test, EM / F1 unless noted)**:

| Method | Backbone / type | Result | Source |
|---|---|---|---|
| InstructDr, zero-shot | BLIP-2-style 3.4B | 31.9 / 40.2 | arXiv 2401.13313 |
| GPT-4 Turbo Vision + OCR | – | 64.7 ANLS | arXiv 2405.18433 |
| Arctic-TILT 0.8B | encoder-decoder | 55.1 / – | AVIR Table 2 |
| VDocRAG (Phi3V) | visual RAG | 44.2 F1 (760-Q subset) | CVPR 2025, arXiv 2504.09795 |
| FRAG top-2 pages | LLaVA-OV-7B / InternVL2-8B / InternVL2-76B | 51.5/60.7 · 52.9/63.3 · 66.0/75.6 | arXiv 2504.17447 |
| Eagle 2.5-8B | long-context VLM | 63.2 / 72.3 (ANLS 72.7) | arXiv 2504.15271 |
| AVIR-3B | retriever + Qwen2.5-3B-AWQ | 60.3 / 68.9 | arXiv 2601.11976 |
| CogDoc (Qwen2.5-VL-7B SFT+RL) | Acc/F1, GPT-4o extractor | 58.3 / 67.9 (GPT-4.1 61.5/74.7; Claude-3.7 62.9/76.3) | arXiv 2512.12658 |
| Doc-V* GRPO (Qwen2.5-VL-7B) | coarse-to-fine agent | 77.2 F1 | arXiv 2604.13731 |
| HierDoc (Qwen3-VL-8B) | routing framework | 76.45 F1 | arXiv 2607.29638 |
| SlideAgent (GPT-4o) | hierarchical agents, custom metric | 84.9 acc / 90.5 F1 | arXiv 2510.26615 |
| EVisRAG-7B (VisRAG 2.0) | evidence-guided RAG | 82.7 acc / 84.5 F1 | arXiv 2510.09733 |

No official leaderboard exists. MDocAgent, Docopilot and M3DocRAG do not evaluate SlideVQA in their own papers; the M3DocRAG SlideVQA numbers come from third-party re-runs.

### 3.6 Qwen performance

| Model | Metric | Value | Setting | Source |
|---|---|---|---|---|
| Qwen2-VL-7B | accuracy | 19.08 | zero-shot multi-image | Leopard Table 4 (arXiv 2410.01744) |
| ColPali + Qwen2-VL-7B (M3DocRAG) | F1 | 55.7 | RAG | Doc-V* / HierDoc Table 1 |
| ColPali + Qwen2-VL-7B (M3DocRAG) | Acc / F1 | 48.8 / 57.9 | RAG, GPT-4o extractor | CogDoc Table 1 |
| Qwen2.5-VL-7B raw | F1 | 55.2 | all pages | Doc-V* / HierDoc |
| Qwen2.5-VL-7B + RAG top-5 | F1 | 62.9 | RAG | Doc-V* |
| Qwen2.5-VL-7B / 72B-Instruct | Acc / F1 | 38.9/49.8 · 38.8/54.4 | end-to-end, GPT-4o extractor | CogDoc Table 1 |
| Qwen2.5-VL-7B-Instruct | ANLS | 70.33 (DocR1: 71.96) | end-to-end | DocR1 Table 4 |
| Qwen2.5-VL-7B / 32B | Acc / F1 | 62.2/30.2 · 79.9/50.4 | VisRAG 2.0 protocol | arXiv 2510.09733 |
| Qwen2.5-VL-7B / 32B-Instruct | overall / F1 | 79.5/94.3 · 79.2/92.2 | SlideAgent custom metric | arXiv 2510.26615 |
| Qwen2.5-3B-AWQ (AVIR baseline) | EM / F1 | 56.6 / 65.8 | vision only, official script | AVIR Table 2 |
| Qwen3-VL-8B inside HierDoc | F1 | 76.45 | routing | HierDoc Table 1 |

Not found: Qwen-VL v1, raw Qwen3-VL. Qwen2/2.5/3-VL technical reports do not list SlideVQA. Raw Qwen2.5-VL-7B numbers span F1 30–94 across papers because prompting, page budgets, extraction and subsets differ; only AVIR, FRAG and Eagle use the official script on the full 2,215-question test set.

---

## Cross-dataset summary

| | MMLongBench-Doc | MP-DocVQA | SlideVQA |
|---|---|---|---|
| Splits used by pipeline | train (eval-only set) | train / val / test | train / val / test |
| Docs / questions | 135 / 1,091 (HF current) | 5,928 / 46,176 | 2,619 / 14,484 |
| Nature | born-digital PDFs (mostly) | scanned industry docs (JPG) | born-digital slides (PNG/JPG) |
| Pages per doc | avg 47.5, max ~120 | 1–20, avg 8.3 | 20 |
| Official metrics | Acc + F1 (GPT-4o extraction + rules) | ANLS + page accuracy | EM + F1 (QA, evidence, joint) |
| Top public result | Claude 4.5 Opus 61.9 Acc | RealDoc-PageTreeIndex 0.8823 ANLS | Doc-V* 77.2 F1 (no official board) |
| Best Qwen (official protocol) | Qwen3-VL-235B 57.0 Acc | AVIR-Qwen2.5-VL-7B 0.8763 ANLS | AVIR (Qwen2.5-3B) 60.3 EM / 68.9 F1 |
| Licence | CC BY-NC 4.0 | RRC registration terms | NTT evaluation licence |

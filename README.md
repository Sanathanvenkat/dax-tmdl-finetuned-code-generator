# DAX/TMDL Fine-Tuned Code Generator

A LoRA fine-tuned language model that converts MicroStrategy (MSTR) cube SQL patterns into Power BI DAX measures and TMDL table/relationship definitions — built as an accelerator for MSTR-to-Power-BI migration work (see [IntelliConvert](#) for the broader platform this extends).

## Why this project

Enterprise BI migrations from MicroStrategy to Power BI require translating hundreds of cube metrics and schema definitions into DAX and TMDL. Doing this via a general-purpose LLM API call per conversion works, but is costly at scale and not ideal for client environments with data residency constraints (e.g. on-prem/VDI setups where calling an external API isn't an option). This project explores whether a small, locally-runnable model can be fine-tuned to handle this specific conversion task directly.

## Approach

Instead of full fine-tuning (expensive, needs large GPU memory, easy to overfit on small data), this uses **QLoRA** — Quantized Low-Rank Adaptation — which freezes the base model entirely and trains a small set of adapter weights on top.

- **Base model:** Qwen2.5-Coder-3B-Instruct
- **Method:** QLoRA (4-bit quantized base + LoRA adapters, r=16, alpha=32)
- **Trainable parameters:** 7.37M out of 3.09B total (**0.24%** of the full model)
- **Training infra:** Google Colab free tier (T4 GPU)
- **Dataset:** 107 hand-seeded + synthetically expanded (instruction, input, output) pairs across 6 conversion pattern categories

## Dataset

| Category | Description |
|---|---|
| Simple aggregation | SUM, COUNT, AVG, MIN, MAX metric conversions |
| Calculated member | Ratio/derived metrics → DAX measures with DIVIDE |
| Filter/conditional | WHERE-clause metrics → CALCULATE/FILTER patterns |
| Table definition | MSTR attribute schemas → TMDL table blocks |
| Relationship | MSTR schema joins → TMDL relationship definitions |
| Time intelligence | YTD/QTD/MTD/MoM/YoY patterns → DAX time functions |

83 train / 6 validation / 18 test examples, split with per-category stratification to ensure every category is represented in each split.

*Current dataset is synthetically generated (LLM-assisted) with hand-seeded real patterns as few-shot anchors. It does not yet include multi-table joins, aliases, or subqueries seen in complex real-world MSTR cube SQL — noted as a scope limitation below.*

## Training results

| Epoch | Train Loss | Val Loss | Token Accuracy |
|---|---|---|---|
| 1 | 1.93 | 1.56 | 70.5% |
| 2 | 1.04 | 0.72 | 84.1% |
| 3 | 0.55 | 0.54 | 87.5% |

Both train and validation loss decreased together across all 3 epochs, indicating genuine learning rather than overfitting on the training set.

## Evaluation: base model vs. fine-tuned vs. fine-tuned + RAG

Evaluated on 18 held-out test examples never seen during training. Each example was run through three configurations: the plain base model (LoRA adapter temporarily disabled via `disable_adapter()` for a true apples-to-apples comparison), the fine-tuned model alone, and the fine-tuned model with retrieval-augmented generation (RAG) grounding it in a schema corpus built from the project's own table/relationship definitions.

| Metric | Base Model | Fine-Tuned | Fine-Tuned + RAG |
|---|---|---|---|
| Exact match | 1/18 (6%) | 4/18 (22%) | **6/18 (33%)** |
| Structural match* | 5/18 (28%) | 14/18 (78%) | 14/18 (78%) |
| Output format validity** | 0/18 (0%) | 18/18 (100%) | 18/18 (100%) |

\* *Structural match: a rule-based check confirming the output uses the correct DAX/TMDL functions and keywords for the target pattern (≥70% overlap with expected structural signals), not a full semantic equivalence check.*

\*\* *Format validity: output is directly usable DAX/TMDL with no markdown code fences, no explanatory prose, and no invented alternate formats (e.g. JSON instead of TMDL).*

### What this shows — and what each technique is actually responsible for

The base model frequently produces technically-plausible-looking DAX but wraps it in markdown fences, adds multi-paragraph explanations, or substitutes an invented JSON schema format instead of TMDL — none of which is directly usable in a migration pipeline without post-processing. It occasionally degenerates into repetition loops on longer outputs.

**Fine-tuning's contribution** is clearest in format validity and structural correctness: 0% → 100% format validity, 28% → 78% structural match. This is the model learning *output discipline and pattern-following* — stop when the answer's done, use the right function shapes.

**RAG's contribution** shows specifically in exact match: 22% → 33%, with the clearest gain on `calculated_member` examples (0/3 → 2/3 exact), where retrieval grounds the model in the correct table and column names instead of it guessing from training patterns alone. Structural match and format validity did not move further with RAG (both stayed flat vs. fine-tuning alone) — RAG's job here is factual grounding on specific identifiers, not format or pattern-following, and the numbers reflect that division of labor rather than RAG being a blanket improvement.

One category (`table_definition`) saw exact match dip slightly with RAG (1/3 → 0/3) — with only 3 examples per category, this is a single example flipping, not a robust regression signal, but noted here rather than hidden.

### Limitations

- Test set is small (n=18, only 3 per category); results are directional, not statistically robust.
- Structural match is a heuristic (keyword/function overlap), not a full DAX/TMDL parser or semantic validator.
- Dataset does not yet cover complex real-world patterns (multi-table joins, aliased subqueries).
- RAG corpus (35 documents) is built from the project's own table/relationship definitions, not an independent held-out client schema — a stronger future test would evaluate retrieval against schema documents the model has never seen in any form during training or corpus construction.

## Project structure

```
├── seed_pairs.json              # 48 hand-written seed examples
├── synthetic_batch1.json        # 59 synthetically expanded examples
├── training_data.json           # combined dataset (107 pairs)
├── train.json / val.json / test.json   # stratified splits
├── train_qlora.py               # QLoRA training script (Colab)
├── eval_generate.py             # base vs. fine-tuned generation script (Colab)
├── schema_corpus.json           # 35-document schema corpus for RAG retrieval
├── rag_pipeline.py              # embedding, vector store, retrieval, RAG-augmented generation (Colab)
├── score_results_v2.py          # generic scorer: exact match, structural match, format validity (2-way or 3-way)
├── eval_results_raw.json        # raw outputs, base vs. fine-tuned
├── eval_results_three_way.json  # raw outputs, base vs. fine-tuned vs. fine-tuned+RAG
└── eval_scored_output.json      # computed metrics by category
```

## Stack

`transformers` · `peft` · `bitsandbytes` · `trl` · `datasets` · `sentence-transformers` · `chromadb` · Qwen2.5-Coder-3B-Instruct · all-MiniLM-L6-v2 · Google Colab (T4)

## Future work

- Expand RAG corpus with an independent, held-out client schema (current corpus is built from the training data's own schema definitions)
- Expand dataset with complex multi-table/aliased patterns
- GGUF export for local CPU inference
- Larger held-out test set for more statistically robust eval
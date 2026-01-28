# Domain-Specific Machine Translation — English → Dutch (Software Domain)

This project implements and evaluates **domain-adaptation pipelines** for English→Dutch machine translation, targeting **software and technical localization content**.

Two complementary paradigms are explored:

1. **Encoder–decoder neural MT** (classical domain fine-tuning)
2. **Decoder-only large language model adaptation** (LoRA instruction tuning)

Both approaches are evaluated on:
- a **general-domain benchmark** (FLORES-200 devtest)
- a **software-domain dataset** (provided)

The goal is to demonstrate **end-to-end MT engineering competence**, including data ingestion, training pipelines, evaluation methodology, and analysis.

---

## 📂 Repository Structure

challenge_1_mt/

│

├── data/

│ ├── raw/

│ │ ├── flores200_dataset/

│ │ └── Dataset_Challenge_1.xlsx

│ └── processed/

│ ├── flores_en_nl/

│ ├── software_mt/

│ └── software_instruct/

│

├── training/

│ ├── encdec_train.py

│ ├── build_instruction_dataset.py

│ └── deconly_lora_train.py

│

├── evaluation/

│ ├── utils.py

│ ├── run_encdec_baseline.py

| ├── run_encdec_finetuned.py

| ├── run_decoder_only_LoRA_finetuned.py

│ ├── run_decoder_only_baseline.py

│ └── aggregate_and_visualize.py

│

├── results/

│ ├── *_predictions.csv

│ ├── *_metrics.csv

│ ├── metrics_summary.csv

│ ├── bleu_comparison.png

│ ├── chrf_comparison.png

│ └── domain_shift.png

│

├── report.md

└── README.md



---

## 🎯 Objectives

- Design a **software-domain fine-tuning pipeline** for a small encoder–decoder Transformer.
- Implement a **decoder-only LoRA instruction-tuning pipeline**.
- Evaluate both on:
  - general-domain text (FLORES-200)
  - software-domain text (provided dataset)
- Report quantitative and qualitative analysis.

---

## 📊 Datasets

### General domain
- FLORES-200 devtest (English–Dutch)

### Software domain
- Provided Excel dataset (UI strings, technical/system messages)

### Training corpus
- OPUS-100 (en–nl)

All datasets are normalized into Hugging Face’s standard `translation` schema.

---

## 🤖 Models

### Encoder–decoder
- `Helsinki-NLP/opus-mt-en-nl`
- MarianMT architecture

### Decoder-only
- `Qwen/Qwen2.5-3B-Instruct`
- QLoRA + instruction tuning

---

## ⚙️ Setup

### Environment

- pip install -U transformers datasets sacrebleu evaluate sentencepiece pandas openpyxl \
- peft bitsandbytes accelerate pytorch-lightning matplotlib


## 🧱 Data preparation
python data/prepare_software_dataset.py
python data/prepare_flores.py
python training/build_instruction_dataset.py

## 🧪 Baseline evaluation

### Encoder–decoder baseline:

python evaluation/run_encdec_baseline.py

### Decoder-only baseline:

python evaluation/run_decoder_only_baseline.py

## 🏗️ Training pipelines

### Encoder–decoder domain fine-tuning
python training/encdec_train.py

### Features:

- OPUS-100 training corpus

- domain prefix tokens

- BLEU-based validation

- mixed precision

### Decoder-only LoRA instruction tuning
python training/deconly_lora_train.py

### Features:

- instruction formatting

- 4-bit quantization

- LoRA adapters

- software-domain specialization

## 📈 Metrics aggregation & visualization

python evaluation/aggregate_and_visualize.py

### Generates:

- results/metrics_summary.csv

- BLEU comparison plot

- chrF++ comparison plot

- Domain-shift analysis plot

### 📄 Final report

report.md

### Includes:
- methodology

- results

- graphs

- analysis

- limitations

## 🧠 Key Engineering Highlights

- Dual MT paradigms (classical NMT + LLMs)

- Domain adaptation strategies

- Instruction tuning

- Low-VRAM LoRA setup

- General vs in-domain evaluation

- Automated metrics + visualizations

- Reproducible pipelines

## ⚠️ Notes

- Fine-tuning is intentionally lightweight to fit time and compute constraints.

- The primary objective is pipeline correctness, evaluation design, and domain analysis.

- The provided dataset is used strictly for in-domain evaluation.

## ✅ Deliverables

- Full data processing pipeline

- Two fine-tuning architectures

- Evaluation harness

- Visualization and reporting layer

- Reproducible research-style project layout

# 👤 Author

## Pruthvish Eshwar


 




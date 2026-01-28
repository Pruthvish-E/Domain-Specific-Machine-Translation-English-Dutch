import torch
import pytorch_lightning as pl
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import evaluate

MODEL_NAME = "Helsinki-NLP/opus-mt-en-nl"
BATCH_SIZE = 8
LR = 2e-5
MAX_LEN = 256
NUM_WORKERS = 2

SOFTWARE_EVAL_SET = "data/processed/software_mt"


class MTModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bleu = evaluate.load("sacrebleu")

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = self.model.generate(batch["input_ids"], max_new_tokens=128)
        pred_text = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_text = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        self.bleu.add_batch(predictions=pred_text, references=[[r] for r in ref_text])
        self.log("val_loss", outputs.loss, prog_bar=True)

    def on_validation_epoch_end(self):
        bleu = self.bleu.compute()["score"]
        self.log("val_bleu", bleu, prog_bar=True)
        self.bleu = evaluate.load("sacrebleu")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)


def prepare_dataset(tokenizer, split="train"):
    """
    Lightweight domain-adaptation recipe:
    - General MT data (e.g., WMT / OPUS)
    - Optional mixing with software-style corpora
    """

    print("Loading opus100 en-nl...")
    ds = load_dataset("opus100", "en-nl", split=split)

    def preprocess(batch):
        src_texts = ["<software> " + x["en"] for x in batch["translation"]]
        tgt_texts = [x["nl"] for x in batch["translation"]]

        model_inputs = tokenizer(
            src_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.shuffle(seed=42).select(range(30000 if split == "train" else 2000))
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)

    return ds


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = prepare_dataset(tokenizer, "train")
    val_ds = prepare_dataset(tokenizer, "validation")

    collator = DataCollatorForSeq2Seq(tokenizer, model=MODEL_NAME)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collator)

    model = MTModule()

    use_cuda = torch.cuda.is_available()

    trainer = pl.Trainer(
        accelerator="cuda" if use_cuda else "cpu",
        devices=1,
        precision="16-mixed" if use_cuda else 32,
        max_epochs=1,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        default_root_dir="models/encdec",
    )

    trainer.fit(model, train_loader, val_loader)
    model.model.save_pretrained("models/encdec/software_mt")
    tokenizer.save_pretrained("models/encdec/software_mt")


if __name__ == "__main__":
    main()

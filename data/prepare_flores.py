from datasets import Dataset
from pathlib import Path

FLORES_ROOT = "data/raw/flores200_dataset/devtest"
SRC_FILE = "eng_Latn.devtest"
TGT_FILE = "nld_Latn.devtest"

OUTPUT_PATH = "data/processed/flores_en_nl"

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines()]

def main():
    src_path = Path(FLORES_ROOT) / SRC_FILE
    tgt_path = Path(FLORES_ROOT) / TGT_FILE

    assert src_path.exists(), f"Missing: {src_path}"
    assert tgt_path.exists(), f"Missing: {tgt_path}"

    en_lines = read_lines(src_path)
    nl_lines = read_lines(tgt_path)

    assert len(en_lines) == len(nl_lines), "Source and target size mismatch"

    data = {
        "translation": [
            {"en": en, "nl": nl}
            for en, nl in zip(en_lines, nl_lines)
        ]
    }

    ds = Dataset.from_dict(data)

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(OUTPUT_PATH)

    print("Saved FLORES devtest EN–NL")
    print("Samples:", ds[0], ds[1], ds[2])

if __name__ == "__main__":
    main()

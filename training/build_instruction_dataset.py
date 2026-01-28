from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path

SOFTWARE_DATASET = "data/processed/software_mt"
OUTPUT_PATH = "data/processed/software_instruct"

SYSTEM_PROMPT = "You are a professional software localization engine. Translate accurately from English to Dutch."

def format_example(en, nl):
    return f"""### System:
{SYSTEM_PROMPT}

### Instruction:
Translate the following software-related text from English to Dutch.

### Input:
{en}

### Response:
{nl}
"""

def main():
    ds = load_from_disk(SOFTWARE_DATASET)

    def convert(example):
        return {
            "text": format_example(
                example["translation"]["en"],
                example["translation"]["nl"]
            )
        }

    ds = ds.map(convert, remove_columns=ds.column_names)

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(OUTPUT_PATH)

    print("Saved instruction dataset to", OUTPUT_PATH)
    print(ds[0]["text"])

if __name__ == "__main__":
    main()

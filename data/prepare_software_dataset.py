import pandas as pd
from datasets import Dataset
from pathlib import Path

INPUT_PATH = "data/raw/Dataset_Challenge_1.xlsx"
OUTPUT_PATH = "data/processed/software_mt"

def main():
    df = pd.read_excel(INPUT_PATH)

    df = df.rename(columns={
        "English Source": "en",
        "Reference Translation": "nl"
    })

    df = df[["en", "nl"]].dropna()

    dataset = Dataset.from_pandas(df)

    dataset = dataset.map(lambda x: {
        "translation": {
            "en": x["en"].strip(),
            "nl": x["nl"].strip()
        }
    })

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(OUTPUT_PATH)

    print("Saved software dataset to:", OUTPUT_PATH)
    print(dataset[0])

if __name__ == "__main__":
    main()

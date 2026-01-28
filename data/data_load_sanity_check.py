from datasets import load_from_disk

soft = load_from_disk("data/processed/software_mt")
flores = load_from_disk("data/processed/flores_en_nl")

print("Software:", soft[0])
print("Flores:", flores[0])
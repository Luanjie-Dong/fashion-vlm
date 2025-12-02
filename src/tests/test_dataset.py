import sys
sys.path.append("../")
from training.dataloader import load_dataset

train, _, _ = load_dataset()
print("Dataset len:", len(train))
sample = train[0]
print("Keys:", sample.keys())
print("Image type:", type(sample["image"]))
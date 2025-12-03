from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split



class FashionDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,
            "prompt": row["input"],
            "target": row['output']
        }
    
class CustomCollator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        return train_collate_fn(
            batch, 
            self.processor, 
            max_length=self.max_length
        )
    
def train_collate_fn(examples,processor,max_length=512):
    images = [example["image"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    labels = [example["target"] for example in examples]


    batch = processor(images=images,text=prompts, suffix=labels, padding=True,     
        truncation="only_second", max_length=max_length,return_tensors="pt")
    
    batch['answers'] = labels
    return batch


def eval_collate_fn(examples,processor,max_length=512):
    images = [example["image"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    answers = [example["target"] for example in examples]

    inputs = processor(text=prompts, images=images, return_tensors="pt")

    return inputs , answers
    


def load_dataset():
    img_dir = "../../data/fashion-data/images"
    df = pd.read_parquet("../../data/vlm_data.pq")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(df,test_size=0.2,random_state=42,shuffle=False)
    val_df, test_df = train_test_split(temp_df,test_size=0.5,random_state=42,shuffle=False)

    #reduce size of val and test for faster training & eval on a single gpu
    val_df = val_df.sample(n=10,random_state=42)
    test_df = test_df.sample(n=10,random_state=42)

    train_dataset , val_dataset , test_dataset = FashionDataset(train_df,img_dir), FashionDataset(val_df,img_dir), FashionDataset(test_df,img_dir)

    print(f"Total: {len(df):,}")
    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")

    return (
        FashionDataset(train_df, img_dir),
        FashionDataset(val_df, img_dir),
        FashionDataset(test_df, img_dir),
    )
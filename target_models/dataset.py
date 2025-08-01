import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CheXpertDataset(Dataset):
    def __init__(self, df, image_root, label_cols, transform=None):
        self.df = df
        self.image_root = image_root
        self.transform = transform
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['Path'])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row[self.label_cols].values.astype("float32"))

        # Extract/store age and sex
        age = torch.tensor(row["Age"], dtype=torch.float32)
        sex = 1.0 if row["Sex"] == "Male" else 0.0
        sex = torch.tensor(sex, dtype=torch.float32)

        return img, label, age, sex

class CheXpertAgeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]["Path"])
        image = Image.open(img_path).convert("L")

        age = self.df.iloc[idx]["Age"]
        age = torch.tensor(age, dtype=torch.float32)
        sex = 1.0 if self.df.iloc[idx]["Sex"] == "Male" else 0.0
        sex = torch.tensor(sex, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, age, sex
    

class CheXpertBinary(Dataset):
    def __init__(self, df, image_root, label_cols, transform=None):
        self.df = df
        self.image_root = image_root
        self.transform = transform
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['Path'])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row.label), dtype=torch.long)

        # Extract/store age and sex
        age = torch.tensor(row["Age"], dtype=torch.float32)
        sex = 1.0 if row["Sex"] == "Male" else 0.0
        sex = torch.tensor(sex, dtype=torch.float32)

        return img, label, age, sex
    
class FeatsDataset(Dataset):
    def __init__(self, json_data, image_root, split="train", transform=None):
        self.data = [item for item in json_data if item["Split"] == split]
        self.root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        entry = self.data[index]

        img_path = os.path.join(self.root, entry["Path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(entry["Label"], dtype=torch.float32)
        age = torch.tensor(entry["Age"], dtype=torch.float32)
        sex = torch.tensor(1.0 if entry["Sex"] == "Male" else 0.0, dtype=torch.float32)

        return image, label, age, sex
from PIL import Image
import torch
from torch.utils.data import Dataset

class DemographicDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.copy()
        self.transform = transform
        self.samples = df.to_dict(orient='records')

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.samples[idx]

        # load image
        image = Image.open(row["Path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # labels
        sex = torch.tensor(row["Patient Sex"]).float()
        age_bin = torch.tensor(row["AgeBin"]).long()
        age = torch.tensor(row["Patient Age"]).float()

        return image, age_bin, sex, age
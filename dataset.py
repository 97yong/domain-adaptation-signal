import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).unsqueeze(1).float()
        self.y = torch.argmax(torch.tensor(y), 1).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(device), self.y[idx].to(device)

def get_dataloaders(opt, source_X_path, target_X_path, target_X_test_path, source_Y_path, target_Y_path, target_Y_test_path):

    source_X = np.load(source_X_path)
    source_Y = np.load(source_Y_path)

    target_X = np.load(target_X_path)
    target_Y = np.load(target_Y_path)

    target_X_test = np.load(target_X_test_path)
    target_Y_test = np.load(target_Y_test_path)

    source_dl = DataLoader(MyDataset(source_X, source_Y), batch_size=opt.batch_size, shuffle=True)
    target_dl = DataLoader(MyDataset(target_X, target_Y), batch_size=opt.batch_size, shuffle=True)
    target_test_dl = DataLoader(MyDataset(target_X_test, target_Y_test), batch_size=opt.batch_size, shuffle=False)
    
    return source_dl, target_dl, target_test_dl

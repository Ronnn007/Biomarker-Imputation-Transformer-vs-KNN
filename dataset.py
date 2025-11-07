import torch
from torch.utils.data import Dataset



class BiomarkerDataset(Dataset):
    def __init__(self, masked_sequences, targets, mask=None):
        self.x_masked = masked_sequences
        self.targets = targets
        self.mask = mask

    def __len__ (self):
        return len(self.x_masked)
    
    def __getitem__(self, index):
        
        if self.mask is not None:
            masked = torch.tensor(self.x_masked[index], dtype=torch.float32)
            target = torch.tensor(self.targets[index], dtype=torch.float32)
            mask_tensor = torch.tensor(self.mask[index], dtype=torch.bool)

            return masked, target, mask_tensor
        
        else:
            
            masked = torch.tensor(self.x_masked[index], dtype=torch.long)
            target = torch.tensor(self.targets[index], dtype=torch.long)

            return masked, target
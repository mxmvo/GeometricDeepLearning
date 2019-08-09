import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



class BodyDataset(Dataset):
    """Body Dataset."""

    def __init__(self, g_files, c_files, transform=None):
        """
        Args:
            g_files: list of files containing the geometry functions
            c_files: list of files containing the connection matrices, these have to be in the same order as the g_files.
        """
        self.g_files = g_files
        self.c_files = c_files
        
        self.transform = transform

    def __len__(self):
        return len(self.g_files)

    def __getitem__(self, idx):
        subject = idx // 10
        pose = idx % 10
        r_pose = np.random.choice(list({i for i in range(10)}- {pose}))
        
        
        r_idx = subject+r_pose
            
        g_1 = torch.load(self.g_files[idx])
        g_2 = torch.load(self.g_files[r_idx])
        
        #i_1, v_1, s_1 = self.sparsetensor(sparse.load_npz(self.c_files[idx]))
        #i_2, v_2, s_2  = self.sparsetensor(sparse.load_npz(self.c_files[r_idx]))
        
        dic_1 = torch.load(self.c_files[idx])
        dic_2 = torch.load(self.c_files[r_idx])
        
        return (g_1, dic_1), (g_2, dic_2)
    
    def sparsetensor(self,sparse_mat):
        
        i = torch.LongTensor([sparse_mat.row,sparse_mat.col])
        v = torch.FloatTensor(sparse_mat.data)
        s = torch.Size(sparse_mat.shape)
        #sparse_tensor = torch.sparse.FloatTensor(i, v, s)
        return i, v, s

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sparse



class BodyDataset(Dataset):
    """Body Dataset."""

    def __init__(self, g_files, c_files, p_files, samples, range_list = list(range(6890)), transform=None):
        """
        Args:
            g_files: list of files containing the geometry functions
            c_files: list of files containing the connection matrices, these have to be in the same order as the g_files.
        """
        self.g_files = g_files
        self.c_files = c_files
        self.p_files = p_files
        self.samples = samples
        self.range_list = range_list

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
        
        points = sparse.load_npz(self.p_files[r_idx])
        
        ind, pos, neg = self.sampling(points)
        

        dic_1 = torch.load(self.c_files[idx])
        dic_2 = torch.load(self.c_files[r_idx])
        
        return (g_1, dic_1), (g_2, dic_2), (ind, pos, neg)
    
    def sampling(self,points):
        indices = np.random.choice(self.range_list, size = self.samples, replace = False)
        pos = np.zeros_like(indices)
        neg = np.zeros_like(indices)

        for i, ind in enumerate(indices):
            row = points[ind].todense()
            pos[i] = np.random.choice(np.where(row == 1)[1], size = 1 )
            neg[i] = np.random.choice(np.where(row == 0)[1], size = 1 )

        return torch.LongTensor(indices), torch.LongTensor(pos), torch.LongTensor(neg)

    def sparsetensor(self,sparse_mat):
        
        i = torch.LongTensor([sparse_mat.row,sparse_mat.col])
        v = torch.FloatTensor(sparse_mat.data)
        s = torch.Size(sparse_mat.shape)
        #sparse_tensor = torch.sparse.FloatTensor(i, v, s)
        return i, v, s

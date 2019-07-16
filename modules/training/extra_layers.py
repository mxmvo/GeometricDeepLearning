import torch
import torch.nn as nn
import torch.nn.functional as F

class GeodesicLayer(nn.Module):
    '''
    An implementation of the Geodesic Convolution Layer. 
    Right now is uses a max pooling over the rotational bins. 
    ''' 
    def __init__(self, in_channels, out_channels, weights = None, p_bins = 5, t_bins = 16, device = 'cpu'):
        super(GeodesicLayer,self).__init__()
        
        self.B = p_bins*t_bins
        self.p = p_bins
        self.t = t_bins
        self.out = out_channels
        self.inp = in_channels
        self.device = device

        if weights is None:
            self.weights = nn.Parameter(torch.randn((self.B, self.inp, self.out), dtype = torch.float))
        else:
            self.weights = nn.Parameter(torch.from_numpy(weights))
       
        self.layer_weights = torch.empty(self.B,self.inp, self.out, self.t)  
        
        for i in range(self.t):
            self.layer_weights[:self.B-self.p*i,:,:,i]  = self.weights[self.p*i:,:,:]
            self.layer_weights[self.B-self.p*i:,:,:,i]  = self.weights[:self.p*i,:,:]


        
        
    def forward(self, x, conn):
        # Make the matrix
        
        self.layer_weights = torch.empty(self.B,self.inp, self.out, self.t)  

        for i in range(self.t):
            self.layer_weights[:self.B-self.p*i,:,:,i]  = self.weights[self.p*i:,:,:]
            self.layer_weights[self.B-self.p*i:,:,:,i]  = self.weights[:self.p*i,:,:]
        
            
        
        self.layer_weights = self.layer_weights.reshape(self.inp*self.B,self.out*self.t)
        self.layer_weights = self.layer_weights.to(self.device)
        # TODO Implement

        x = torch.sparse.mm(conn, x)
        x = x.reshape(-1,self.B*self.inp)
        x = torch.matmul(x, self.layer_weights)

        x = x.reshape(-1,self.out,self.t)
        return torch.max(x, dim = -1)[0]

    
class EquivariantLayer(nn.Module):
    '''
    An implementation of the Geodesic Convolution Layer. 
    Right now is uses a max pooling over the rotational bins. 
    ''' 
    def __init__(self, C_in, C_out, R_in, R_out, weights = None, p_bins = 5, t_bins = 16, device = 'cpu'):
        super(EquivariantLayer,self).__init__()
        
        self.B = p_bins*t_bins
        self.p = p_bins
        self.t = t_bins
        self.C_out = C_out
        self.C_in = C_in
        self.R_in = R_in
        self.R_out = R_out
        self.device = device


        if weights is None:
            self.weights = nn.Parameter(torch.randn((self.C_in*self.R_in * self.B, self.C_out), dtype = torch.float).to(self.device))
        else:
            self.weights = nn.Parameter(torch.from_numpy(weights).float()).to(self.device)
       
        x = torch.arange(0,self.R_in*self.B)
        self.rotation_matrix = self.rotate(x)

        self.index_sparse = self.make_index_sparse(self.rotation_matrix)

        self.update_layer_weights()
        '''
        self.index_matrix = torch.zeros((self.R_out,self.C_in * self.R_in * self.B, self.C_in * self.R_in * self.B), dtype = torch.float)

        for i in range(self.R_out):
            self.index_matrix[i] = self.index_matrix[i].scatter(0,self.rotation_matrix[:,i].view(1,-1), 1)

        self.index_matrix = self.index_matrix.to(self.device)
        '''
        #self.layer_weights = torch.empty(self.C_in*self.R_in*self.B,self.C_out*self.R_out).to(self.device) 


    def make_index_sparse(self,R):
        cols = R.reshape(-1)
        rows = torch.arange(0,len(cols))
        data = torch.tensor([1]*len(cols))

        i = torch.LongTensor([rows.data.numpy(),cols.data.numpy()])
        v = data.float()
        index_sparse = torch.sparse.FloatTensor(i, v, torch.Size([self.R_in*self.B*self.C_in*self.R_out,self.R_in*self.B*self.C_in])).to(self.device)
        return index_sparse

    def update_layer_weights(self):
        l_weights = torch.sparse.mm(self.index_sparse, self.weights)
        self.l_weights = l_weights.reshape(self.C_in,self.R_in,self.B,self.R_out,self.C_out).permute(2,0,1,4,3).reshape(self.C_in*self.R_in*self.B,-1)


        
    def forward(self, x, conn):
        # Make the matrix
        #self.layer_weights = torch.empty(self.C_in*self.R_in*self.B,self.C_out*self.R_out).to(self.device) 
        
        #l_weights = torch.matmul(self.weights.unsqueeze(1).unsqueeze(1),self.index_matrix)
        #l_weights = l_weights.squeeze().reshape(-1,self.C_in * self.R_in * self.B).t()
        #l_weights = torch.sparse.mm(self.index_sparse, self.weights)
        #l_weights = l_weights.reshape(-1,self.R_out,self.C_out).permute(0,2,1).reshape(self.C_in*self.R_in*self.B,-1)

        x = torch.sparse.mm(conn, x)
        x = x.reshape(-1,self.B*self.C_in*self.R_in)
        x = torch.matmul(x, self.l_weights)
        return x

    def update_layer(self):
        self.update_layer_weights() 
        #for i in range(self.C_out):
        #     pif = self.weights[i][self.rotation_matrix]
        #     self.layer_weights[:,i*self.R_out:(i+1)*self.R_out] = pif

        

    def angle_rotation(self,x):
        pif = x.view(-1, self.p)

        pif_1 = torch.zeros(pif.shape, dtype = torch.int)

        for i in range(self.R_in):
            pif_1[i*self.t:(i+1)*self.t-1,:] = pif[i*self.t+1:(i+1)*self.t,:]
            pif_1[(i+1)*self.t-1,:] = pif[i*self.t,:]
        
        return pif_1.view(-1)

    def kernel_rotation(self,x):
        pif = x.view(-1,self.t*self.p)
        pif_1 = torch.zeros(pif.shape, dtype = torch.int)

        pif_1[1:,:] = pif[:-1,:]
        pif_1[0,:] = pif[-1,:]
        
        return pif_1.view(-1)

    def rotate(self,x):
        kernel_small = torch.zeros((self.R_in*self.t*self.p, self.R_out), dtype =  torch.long)

        pif = x.clone()
        kernel_small[:,0] = pif
        
        for j in range(1,self.R_out):
            kernel_small[:,j] = self.kernel_rotation(self.angle_rotation(pif))
            pif = kernel_small[:,j]
            
        kernel = torch.zeros((self.C_in*self.R_in*self.t*self.p,self.R_out), dtype = torch.long)
        
        for i in range(self.C_in):
            kernel[i*self.R_in*self.t*self.p:(i+1)*self.R_in*self.t*self.p] = kernel_small + i*self.R_in*self.t*self.p
        return kernel


class AMP(nn.Module):
    '''
    An implementation of the Geodesic Convolution Layer. 
    Right now is uses a max pooling over the rotational bins. 
    ''' 
    def __init__(self, regular_size = 16):
        super(AMP,self).__init__()
        
        self.R = regular_size
        
    def forward(self, x):
        # Make the matrix
        N = x.shape[0]
        res = x.view(-1,self.R)
        
        res = torch.max(res, dim = -1)[0]
        return res.view(N, -1)
    

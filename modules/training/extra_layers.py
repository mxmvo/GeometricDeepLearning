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
            self.weights = nn.Parameter(torch.randn((self.C_out,self.C_in*self.R_in * self.B), dtype = torch.float))
        else:
            self.weights = nn.Parameter(torch.from_numpy(weights))
       
        x = torch.arange(0,self.R_in*self.B)
        self.rotation_matrix = self.rotate(x)

        
        self.layer_weights = torch.empty(self.C_in*self.R_in*self.B,self.C_out*self.R_out)  

        for i in range(self.C_out):
            pif = self.weights[i][self.rotation_matrix]
            self.layer_weights[:,i*self.R_out:(i+1)*self.R_out] = pif


        
        
    def forward(self, x, conn):
        # Make the matrix
        
        self.layer_weights = torch.empty(self.C_in*self.R_in*self.B,self.C_out*self.R_out)  

        for i in range(self.C_out):
            pif = self.weights[i][self.rotation_matrix]
            self.layer_weights[:,i*self.R_out:(i+1)*self.R_out] = pif
        
        self.layer_weights = self.layer_weights.to(self.device)
        x = torch.sparse.mm(conn, x)
        x = x.reshape(-1,self.B*self.C_in*self.R_in)
        x = torch.matmul(x, self.layer_weights)
        return x
        

    def angle_rotation(self,x):
        pif = x.view(-1, self.p)

        pif_1 = torch.zeros(pif.shape, dtype = torch.int)

        for i in range(self.R_in):
            pif_1[i*self.t:i*self.t+self.t-1,:] = pif[i*self.t+1:i*self.t+self.t,:]
            pif_1[i*self.t+self.t-1,:] = pif[i*self.t,:]
        
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
    def __init__(self, regular_size = 80):
        super(AMP,self).__init__()
        
        self.R = regular_size
        
    def forward(self, x):
        # Make the matrix
        N = x.shape[0]
        res = x.view(-1,self.R)
        
        res = torch.max(res, dim = -1)[0]
        return res.view(N, -1)
    

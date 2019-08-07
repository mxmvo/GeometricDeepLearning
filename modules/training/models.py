import torch
import torch.nn as nn
import torch.nn.functional as F

import glob, re
import numpy as np

from modules.training.extra_layers import GeodesicLayer, EquivariantLayer, AMP


class GCCN_1(nn.Module):
    def __init__(self, neurons = [], weights = None, p_bins = 5, t_bins = 16):
        super(GCCN_1,self).__init__()
        self.linear_1 = nn.Linear(neurons[0], neurons[1])
        self.GC_1 = GeodesicLayer(neurons[1], neurons[2])
        
    
    def forward(self, x, conn):
        res = F.relu(self.linear_1(x))
        res = self.GC_1(res, conn)
        return res.squeeze()
    
    def load_model(self, params, it = None):
        try:
            if it == None:
                models = sorted(glob.glob(params['model_dir']+'*.mdl'))
            
                I = [ int(re.findall(r'_([0-9]+).mdl', i)[0]) for i in models]
                m = models[np.argmax(I)]
            else:
                models = glob.glob(params['model_dir']+'*'+str(it)+'.mdl')
                m = models[0]
            
            it = re.findall('_([0-9]+).mdl',m)[0]
            params['it'] = int(it)
            self.load_state_dict(torch.load(m))
            print('Loaded model: {}'.format(m))
        except:
            print('Couldn\'t load model')
            params['it'] = 0
        return params


class GCCN_2(nn.Module):
    def __init__(self, neurons = [], weights = None, p_bins = 5, t_bins = 16, device = 'cpu', l_bias = False):
        super(GCCN_2,self).__init__()
        self.linear_1 = nn.Linear(neurons[0], neurons[1], bias = l_bias)
        self.GC_1 = GeodesicLayer(neurons[1], neurons[2], device = device)
        self.device = device 
    
    def forward(self, x, conn):
        res = F.relu(self.linear_1(x))
        res = self.GC_1(res, conn)
        norm = torch.norm(res.squeeze(), dim = - 1, keepdim = True)
        return res/norm

    def update_layers(self):
        None    
    
    def load_model(self, params, it = None):
        try:
            if it == None:
                models = sorted(glob.glob(params['model_dir']+'*.mdl'))
            
                I = [ int(re.findall(r'_([0-9]+).mdl', i)[0]) for i in models]
                m = models[np.argmax(I)]
            else:
                models = glob.glob(params['model_dir']+'*'+str(it)+'.mdl')
                m = models[0]
            
            it = re.findall('_([0-9]+).mdl',m)[0]
            params['it'] = int(it)
            with open(m, 'rb') as f:
                s_model = torch.load(f, map_location = self.device)
                self.load_state_dict(s_model[0])
            print('Loaded model: {}'.format(m))
        except Exception as e:
            print('Couldn\'t load model', e)
            params['it'] = 0
        return params

class GCCN_3(nn.Module):
    def __init__(self, neurons = [], weights = None, p_bins = 5, t_bins = 16, device = 'cpu', l_bias = False):
        super(GCCN_3,self).__init__()
        self.device = device
        self.linear_1 = nn.Linear(neurons[0], neurons[1], bias = l_bias)
        self.GC_1 = GeodesicLayer(neurons[1], neurons[2], device = device)
        self.GC_2 = GeodesicLayer(neurons[2], neurons[3], device = device)
        
    
    def forward(self, x, conn):
        res = F.relu(self.linear_1(x))
        res = F.relu(self.GC_1(res, conn))
        res = self.GC_2(res,conn)
        norm = torch.norm(res.squeeze(), dim = - 1, keepdim = True)
        return res/norm

    def update_layers(self):
        None

    def load_model(self, params, it = None):
        try:
            if it == None:
                I = [ int(re.findall(r'_([0-9]+).mdl', i)[0]) for i in models]
                m = models[np.argmax(I)]
            else:
                models = glob.glob(params['model_dir']+'*'+str(it)+'.mdl')
                m = models[0]
            
            it = re.findall('_([0-9]+).mdl',m)[0]
            params['it'] = int(it)
            with open(m, 'rb') as f:
                s_model = torch.load(f, map_location = self.device)
                self.load_state_dict(s_model[0])
            print('Loaded model: {}'.format(m))
        except Exception as e:
            print('Couldn\'t load model', e)
            params['it'] = 0
        return params

class GCCN_4(nn.Module):
    def __init__(self, neurons = [], weights = None, p_bins = 5, t_bins = 16, device = 'cpu', l_bias = False):
        super(GCCN_4,self).__init__()
        self.linear_1 = nn.Linear(neurons[0], neurons[1], bias = l_bias)
        self.GC_1 = EquivariantLayer(neurons[1], neurons[2], R_in = 1, R_out = t_bins, device = device)
        self.GC_2 = EquivariantLayer(neurons[2], neurons[3], R_in = t_bins, R_out = t_bins, device = device)
        self.amp = AMP(regular_size = 16)
        
    
    def forward(self, x, conn):
        res = F.relu(self.linear_1(x))
        res = F.relu(self.GC_1(res, conn))
        res = self.GC_2(res,conn)
        res = self.amp(res)
        norm = torch.norm(res.squeeze(), dim = - 1, keepdim = True)
        return res/norm

    def update_layers(self):
        self.GC_1.update_layer()
        self.GC_2.update_layer()

    
    def load_model(self, params, it = None):
        try:
            if it == None:
                models = sorted(glob.glob(params['model_dir']+'*.mdl'))
            
                I = [ int(re.findall(r'_([0-9]+).mdl', i)[0]) for i in models]
                m = models[np.argmax(I)]
            else:
                models = glob.glob(params['model_dir']+'*'+str(it)+'.mdl')
                m = models[0]
            
            it = re.findall('_([0-9]+).mdl',m)[0]
            params['it'] = int(it)
            self.load_state_dict(torch.load(m))
            print('Loaded model: {}'.format(m))
        except:
            print('Couldn\'t load model')
            params['it'] = 0
        return params

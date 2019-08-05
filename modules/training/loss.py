import torch

def siamese_loss(out, out_pos, out_neg,params):
    '''
    Implementation of the siamese loss
    '''
    gamma = params['loss_gamma']
    mu = params['loss_mu']
    pos_term = torch.norm(out-out_pos, dim = -1)**2
    neg_term = torch.norm(out-out_neg, dim = -1)
    
    neg_term = (mu-neg_term)
    neg_term = torch.max(neg_term, torch.Tensor([0]).to(params['device']))**2
    return (1-gamma)*torch.sum(pos_term)+gamma*torch.sum(neg_term)

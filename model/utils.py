import torch
import os
import numpy as np

def to_device(values, device):
    if type(values) is tuple:
        return tuple(to_device(v, device) for v in values)
    if type(values) is list:
        return [to_device(v, device) for v in values]
    if type(values) is torch.Tensor:
        return values.to(device)
    return values


def checkpoint(path, model, epoch, loss, optimizer):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'{path}{os.sep}checkpoint_{epoch:0>3}.pt')
    
    
def load_checkpoint(path, model, optimizer=None):
    if not os.path.exists(path):
        return 0
    file_path = None
    checkpoint_epoch = 0
    for f in os.listdir(path):
        if f.startswith('checkpoint_'):
            epoch = int(f[11:].split('.')[0])
            if epoch > checkpoint_epoch:
                checkpoint_epoch = epoch
                file_path = f'{path}{os.sep}{f}'
    if file_path is not None:
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint_epoch


def interpret_inference(idxs, pred, in_view):
    res = []
    for idx, p, v in zip(idxs, pred, in_view):
        p = p[:v]
        order = np.argsort(-p) 
        pos = [0] * v
        for i, p in enumerate(order, start=1):
            pos[p] = i
        res.append((idx, pos))
    return res


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    pass
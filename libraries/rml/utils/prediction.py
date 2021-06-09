import torch
import torch.utils.data

from .utils import device, tqdm
from typing import Any, Callable

from functools import reduce
from operator import add

def eval(f):
    def wrapper(model: torch.nn.Module, *args, **kwargs):
        model.eval()
        
        with torch.no_grad():
            result = f(model, *args, **kwargs)

        model.train()
        
        return result
    
    return wrapper

def to_device(inp):
    return inp.to(device)

@eval
def predict_one(model: torch.nn.Module, inp, transform: Callable[[Any], torch.FloatTensor]=None, 
                batch_dim: int = 0) -> torch.FloatTensor:
    if transform:
        inp = transform(inp)

    inp = inp.unsqueeze(batch_dim)
    inp = inp.to(device)

    return model(*inp)

@eval
def predict_many(model: torch.nn.Module, inp: Any, bs: int = -1, batch_dim: int = 0,
                 to_device_fn=to_device) -> torch.FloatTensor:
    inp = inp.to(device)
    res = []
    
    for lower in tqdm(range(0, len(inp), bs)):
        batch = inp[lower:min(len(inp), lower + bs)]
        res.append(model(batch))

    return torch.cat(res, dim=batch_dim)

@eval
def predict_dl(model: torch.nn.Module, dl: torch.utils.data.DataLoader, to_device_fn=to_device) -> (torch.FloatTensor, Any):
    pred, targ = [], []
    
    for batch in tqdm(dl):
        inp, target = batch
                
        pred.append(model(*to_device_fn(inp)))
        targ.append(target)
        
    return torch.cat(pred), torch.cat(targ)
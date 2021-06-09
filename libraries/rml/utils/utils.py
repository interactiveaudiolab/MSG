import torch

import tqdm as bar

from typing import Iterable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_notebook = False


def tqdm(x: Iterable, **kwargs) -> bar.tqdm:
    if is_notebook:
        return bar.tqdm_notebook(x, **kwargs)
    else:
        return bar.tqdm(x, **kwargs)


def set_is_notebook(notebook:bool = True) -> None:
    global is_notebook
    
    is_notebook = notebook

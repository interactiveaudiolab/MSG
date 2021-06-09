import torch

import numpy as np
import matplotlib.pyplot as plt
import operator as op

from typing import *
from functools import reduce


def combine_epochs(train_result, key):
    data = []
    
    for tr in train_result:
        data.append(tr[key])
    
    return reduce(op.add, data)  


def simple_smooth(data, factor: int = 5):
    smoother = []

    for i in range(factor, len(data) - factor):
        smoother.append(sum(data[i - factor:i + factor]) / (factor * 2))

    return smoother, np.linspace(factor, len(data) - factor - 1, len(data) - factor * 2)


def plot_fn(f, xmin, xmax, samples=50, name=None, use_torch: bool = False):
    backend = torch if use_torch else np

    xs = backend.linspace(xmin, xmax, samples)
    ys = f(xs)

    if use_torch:
        xs = xs.numpy()
        ys = ys.numpy()

    plt.plot(xs, ys, label=name)


def plot_list(ls: Collection[Any], name: Optional[str] = None, smooth: Optional[int] = None, show: bool = False,
              method=plt.plot) -> None:
    if smooth is None:
        size = len(ls)
        xs = np.linspace(1, size, size)
    else:
        ls, xs = simple_smooth(ls, smooth)

    method(xs, ls, label=name)

    if show:
        if name is not None:
            plt.legend()
        plt.show()


def plot_train_data(data):
    plt.figure(figsize=(15,7))

    train = data["train"]
    train_cost = combine_epochs(train, "cost")

    plot_list(train_cost, smooth=1 + len(train_cost) // 100, name="train")

    if "test" in data:
        test = data["test"]
        test_cost = []

        for te in test:
            test_cost.append(te["cost"])

        test_x = list(map(lambda i: (i + 1) * len(train[0]["cost"]), range(len(test_cost))))

        plt.plot(test_x, test_cost, label="test")

    plt.ylim(ymin=0)
    plt.legend()
    
    
def disp_images(images, rows: int, cols: int, im_width: int = 3, im_height: int = 3):
    f, ax = plt.subplots(rows, cols, figsize=(im_width * cols, im_height * rows))
    
    for row in range(rows):
        for col in range(cols):
            if cols == 1:
                ax[row].imshow(images[row])
            elif rows == 1:
                ax[col].imshow(images[col])
            else:
                idx = col + row * cols
                if idx < len(images):
                    ax[row, col].imshow(images[idx])

    return f, ax
    
    
def plot_lr_range(data, log=True, cutoff: int = -1, smooth: int = None):
    cost = combine_epochs(data["train"], "cost")[:cutoff]
    lrs = combine_epochs(data["train"], "lrs")[:cutoff]
    
    if smooth is not None:
        cost, _ = simple_smooth(cost, smooth)
        lrs = lrs[smooth:-smooth]
    
    if log:
        plt.semilogx(lrs, cost)
    else:
        plt.plot(lrs, cost)

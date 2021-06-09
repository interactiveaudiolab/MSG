import torch

from typing import Union

from functools import reduce
from operator import mul

def save_model(model: torch.nn.Module, path: str) -> None:
    torch.save(model, path)


def load_model(path: str) -> None:
    return torch.load(path)


def save_params(model: torch.nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_params(model: torch.nn.Module, path: str) -> None:
    model.load_state_dict(torch.load(path))


def layer_to_idx(model: torch.nn.Module, layer: str) -> int:
    modules = list(model._modules)

    return modules.index(layer)


def is_frozen(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.requires_grad:
            return False

    return True


def is_unfrozen(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if not param.requires_grad:
            return False

    return True


def freeze(model: torch.nn.Module, do_freeze: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad = not do_freeze


def freeze_model_until(model: torch.nn.Module, layer: Union[str, int], do_freeze: bool = True) -> None:
    if type(layer) == str:
        layer = layer_to_idx(model, layer)

    for layer in list(model.children())[0:layer]:
        freeze(layer, do_freeze)


def freeze_model_after(model: torch.nn.Module, layer: Union[str, int], do_freeze=True) -> None:
    if type(layer) == str:
        layer = layer_to_idx(model, layer)

    for layer in list(model.children())[layer:]:
        freeze(layer, do_freeze)


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def scale_learning_rates(optimizer: torch.optim.Optimizer, scale: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] *= scale


def param_count(model: torch.nn.Module) -> int:
    sizes = [p.size() for p in list(model.parameters())]

    return reduce(lambda acc, x: acc + reduce(mul, x), sizes, 0)

import torch


def accuracy(preds, target) -> float:
    preds = preds.max(-1)[1]
    result = preds == target

    return result.sum().item() / target.size(0)


def accuracy_binary(preds, target, cutoff: float = 0.5) -> float:
    preds = torch.tensor(preds.view(-1))
    preds[preds >= cutoff] = 1
    preds[preds < cutoff] = 0
    result = preds.long() == target
        
    return result.sum().item() / target.size(0)


def precission_recall(preds: torch.FloatTensor, target: torch.LongTensor, cutoff: float = 0.5):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    
    preds = preds.view(-1)
    preds[preds >= cutoff] = 1
    preds[preds < cutoff] = 0
    
    for p, t in zip(preds, target):        
        if p == 1 and t == 1:
            true_pos += 1
        elif p == 1 and t == 0:
            false_pos += 1
        elif p == 0 and t == 1:
            false_neg += 1
        elif p == 0 and t == 0:
            true_neg += 1
    
    return {
        "precission": true_pos / (true_pos + false_pos),
        "recall": true_pos / (true_pos + false_neg),
        "tp": true_pos / len(preds),
        "tn": true_neg / len(preds),
        "fp": false_pos / len(preds),
        "fn": false_neg / len(preds),
    }


def confusion_matrix(preds: torch.FloatTensor, target: torch.LongTensor):
    pass

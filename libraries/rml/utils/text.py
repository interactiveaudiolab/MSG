import os
import shutil
import torch

from torch import nn
from .utils import tqdm

tokenizers = {}


def embedding_from_vocab(vocab, dim=None):
    if vocab.vectors is not None:
        emb = nn.Embedding(len(vocab.itos), emb_dim(vocab), vocab.stoi["<pad>"])        
        emb.load_state_dict({"weight": vocab.vectors}) 
                
        return emb
    else:
        return nn.Embedding(len(vocab.itos), dim, vocab.stoi["<pad>"])


def spacy(lang, spacy_lib):
    if lang not in tokenizers:
        tokenizers[lang] = spacy_lib.load(lang)
        
    tok = tokenizers[lang]
    return lambda text: [tok.text for tok in tok.tokenizer(text)]


def itos_sentence(vocab, sentence):
    return " ".join([vocab.itos[i] for i in sentence])


def itos_sentence_batch(vocab, batch):
    return [itos_sentence(vocab, sentence) for sentence in batch.transpose(0, 1)]


def emb_dim(vocab):
    return vocab.vectors.size()[1]


def seq2seq_nll_loss(ignore_padding=False, vocab=None):
    loss = None
    
    if ignore_padding:
        loss = nn.NLLLoss(ignore_index=vocab.stoi["<pad>"])
    else:
        loss = nn.NLLLoss()
    
    def f(preds, targ):
        if targ.size()[0] < preds.size()[0]:
            targ = nn.functional.pad(targ, (0, 0, 0, preds.size()[0] - targ.size()[0]), value=1)
        elif targ.size()[0] > preds.size()[0]:
            targ = targ[:preds.size()[0]] #for variable length output this may learn to output smaller sequences; less possibilities to make mistakes

        preds = preds.view(-1, preds.size()[-1])
        targ = targ.view(-1)

        return loss(preds, targ)
        
    return f


def merge_hidden_from_layers(hidden):
    bs = hidden.size()[1]
    
    return hidden.transpose(0, 1).contiguous().view(1, bs, -1)


def split_hidden_to_layers(hidden, layer_size):
    _, bs, hidden_size_old = hidden.size()
    hidden_size = hidden_size_old // layer_size
    
    hidden = hidden.view(1, bs, layer_size, hidden_size)
    hidden = hidden.transpose(0, 2).contiguous()
    hidden = hidden.view(layer_size, bs, hidden_size)
    
    return hidden
    
    
def set_fasttext_size(path, language, max_words=None):
    original = os.path.join(path, f"wiki.{language}.vec.original")
    current = os.path.join(path, f"wiki.{language}.vec")
        
    is_new = not os.path.exists(original)
        
    if is_new:
        shutil.move(current, original)
        
    total_size = 0
            
    with open(original) as f:
        total_size = int(f.readline().split(" ")[0])

    current_size = total_size if is_new else 0

    if not is_new:
        with open(current) as f:
            current_size = int(f.readline().split(" ")[0])
        
    if current_size == max_words:
        return
    
    if max_words is None or max_words >= total_size:
        shutil.copy(original, current)
        return

    with open(original, "r") as inp:
        with open(current, "w") as output:
            output.write(str(max_words) + " 300\n")
            next(inp)
            
            for i in tqdm(range(max_words)):
                output.write(inp.readline())


class TextDL():
    def __init__(self, dl):
        self.dl = dl
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        itr = iter(self.dl)
        
        for i in range(len(self.dl)):
            batch = next(itr)
            yield batch.src, batch.trg


def preprocess(sentence, field):
    sentence = field.preprocess(sentence)
    sentence = map(lambda word: field.vocab.stoi[word], sentence)
    
    return torch.LongTensor(list(sentence))
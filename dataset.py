from torch.utils.data import Dataset
import numpy as np


class Tokenizer:
    def __init__(self, length=512):
        self.length = length
        self.bos = 0
        self.eos = 1
        self.pad = 3
        self.special_tokens = [self.bos, self.eos, self.pad]
        self.chars = list(open('data/charmaps.txt').read())
        self.charmap = {k: v + len(self.special_tokens) for v, k in enumerate(self.chars)}
        self.lows = list('abcdefghijklmnopqrstuvwxyz')
        self.ups = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.thas = list('กขคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮะาำเแโใไๅๆ')
        vovels = 'ะัาำิีึืฺุูเแโใไๅๆ็่้๊๋์ํ'
        vovels = list(vovels)
        self.vovels = [vovels[1], *vovels[4:10], *vovels[18:]]

    def __len__(self):
        return len(self.charmap) + len(self.special_tokens)

    def __call__(self, text):
        y = [self.bos]
        x = [self.bos]
        for char in text:
            if char not in self.charmap:
                continue
            if char in self.thas:
                if np.random.rand() > 0.75:
                    x.append(self.charmap[np.random.choice(self.thas)])
                else:
                    x.append(self.charmap[char])
            elif char in self.lows:
                if np.random.rand() > 0.75:
                    x.append(self.charmap[np.random.choice(self.lows)])
                else:
                    x.append(self.charmap[char])
            elif char in self.ups:
                if np.random.rand() > 0.75:
                    x.append(self.charmap[np.random.choice(self.ups)])
                else:
                    x.append(self.charmap[char])
            elif char in self.vovels:
                if np.random.rand() > 0.90:
                    x.append(self.charmap[np.random.choice(self.vovels)])
                elif np.random.rand() > 0.90:
                    pass
                else:
                    x.append(self.charmap[char])
            else:
                x.append(self.charmap[char])
            y.append(self.charmap[char])

        y.append(self.eos)
        x.append(self.eos)
        if len(x) < self.length:
            for i in range(self.length-len(x)):
                x.append(self.pad)
        else:
            x = x[:512]
        if len(y) < self.length:
            for i in range(self.length-len(y)):
                y.append(self.pad)
        else:
            y = y[:512]
        return x, y

    def batch_encode_plus(self, batch):
        out = [self(b) for b in batch]
        return out

    def decode(self, tokens):
        out = ''
        for token in tokens:
            if token in [0, 1]:
                continue
            out += self.chars[token - len(self.special_tokens)]
        return out

    def batch_decode(self, batch):
        out = [self.decode(b) for b in batch]
        return out


class MaskedData(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        x, y = self.tokenizer(tokens)
        return x, y

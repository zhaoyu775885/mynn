import torch
import torch.nn as nn


def log_sum_exp():
    pass

class CRF(nn.Module):
    def __init__(self, ntag, sot, eot):
        super(CRF, self).__init__()
        self.ntag = ntag
        self.eot = eot
        self.sot = sot

        self.trans_mat = nn.Parameter(torch.rand([self.ntag, self.ntag], requires_grad=True))
        self.trans_mat.data[:, self.sot] = -torch.Tensor([float('inf')])
        self.trans_mat.data[self.eot, :] = -torch.Tensor([float('inf')])

    def forward(self, frames):
        for frame in frames:
            emit_score = None
            trans_score = None

    def path_score(self, emits, tags):
        time_len = emits.shape[0]
        indices = torch.LongTensor(range(time_len))
        score = torch.sum(self.trans_mat[tags[:-1], tags[1:]]) + torch.sum(emits[indices, tags])
        return score

    def norm_factor(self, emits):
        time_len = emits.shape[0]
        fwd = torch.zeros(self.ntag)
        for i in range(time_len):
            fwd_expanded = fwd.unsqueeze(1).expand([-1, self.ntag])
            emit_expanded = emits[i].unsqueeze(0).expand([self.ntag, -1])
            tmp = fwd_expanded + self.trans_mat + emit_expanded
            fwd = log_sum_exp(tmp)
        return 

    def decode(self, emits):
        dp_val = torch.zeros([self.ntag, self.ntag], dtype=torch.float)
        dp_idx = torch.zeros([self.ntag, self.ntag], dtype=torch.long)
        time_len = emits.shape[0]
        for i in range(1, time_len):
            tmp = dp_val[:, i-1].unsqueeze(1).expand([-1, self.ntag])
            dp_val[:, i], dp_idx[:, i] = torch.max(tmp+self.trans_mat)
            dp_val[:, i] += emits[i, :]
        final_val, final_argmax = torch.max(dp_val[:, -1])
        backtrack = [final_argmax]
        for i in range(time_len-1, 0, -1):
            final_argmax = dp_idx[final_argmax, i]
            backtrack.append(final_argmax)
        return final_val, backtrack

class RNN_CRF(nn.Module):
    def __init__(self, vocab_size, nemb, nhid, nclass, nlayer=1, bidir=False, dropout=0.2):
        super(RNN_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.nemb = nemb
        self.nhid = nhid
        self.nclass = nclass
        self.nlayer = nlayer
        self.ndir = 2 if bidir else 1
        self.dpr = dropout

        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=nemb)
        self.dropout1 = nn.Dropout(self.dpr)
        self.rnn = nn.GRU(input_size=nemb, hidden_size=nhid, num_layers=nlayer,
                          dropout=0 if nlayer == 1 else dropout,
                          bidirectional=bidir)
        self.dropout2 = nn.Dropout(self.dpr)
        self.fc = nn.Linear(nhid * self.ndir, self.nclass)
        self.crf = CRF(self.nclass)

    def forward(self, x):
        pass

    def decode(self):
        pass

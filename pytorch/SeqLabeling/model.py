import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, nclass):
        super(CRF, self).__init__()
        self.nclass = nclass
        self.trans_mat = nn.Parameter(torch.rand([self.nclass, self.nclass], requires_grad=True))

    def forward(self, frames):
        for frame in frames:
            emit_score = None
            trans_score = None

    def path_score(self, emits, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, emit in enumerate(emits):
            score = score + self.trans_mat[tags[i], tags[i + 1]] + emit[tags[i + 1]]
        score = score + self.trans_mat[self.tag_to_ix[tags[-1], STOP_TAG]]
        return score

    def norm_factor(self, emits):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def decode(self, x):
        pass

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNNaive(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNNaive, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        if hidden.shape[0] < input.shape[0]: hidden = hidden.expand([input.shape[0], self.hidden_size])
        combined = torch.cat((input, hidden), 1)
        hidden = F.relu(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=2,
                batch_first=True, bidirectional=True)
        self.h2o = nn.Linear(self.hidden_size*2, output_size)

    def forward(self, inputs):
        hiddens, _ = self.rnn(inputs)
        #hiddens = F.tanh(hiddens)
        outputs = self.h2o(hiddens)
        return outputs

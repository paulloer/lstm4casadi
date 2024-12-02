"""
The MIT License (MIT)

Copyright (c) 2024 Paul Loer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import casadi as cas
from time import time

def sigmoid(x):
  return 1 / (1 + cas.exp(-x))

class LSTMCasADi:
    def __init__(self, model: nn.Module):
        self.input_size = model.lstm.input_size
        self.num_layers = model.lstm.num_layers
        self.hidden_size = model.lstm.hidden_size
        self.output_size = model.fc.out_features
        assert model.lstm.bidirectional == False
        assert model.lstm.dropout == 0.0
        assert model.lstm.proj_size == 0

        lstm_state_dict = model.lstm.state_dict()
        self.W, self.b = {}, {}
        i_indices = ['ii', 'if', 'ig', 'io']
        h_indices = ['hi', 'hf', 'hg', 'ho']
        for lyr in range(self.num_layers):
            # Extract LSTM weights and biases
            weight_ih = lstm_state_dict[f"weight_ih_l{lyr}"]  # [4*hidden_size, input_size]
            weight_hh = lstm_state_dict[f"weight_hh_l{lyr}"]  # [4*hidden_size, hidden_size]
            bias_ih = lstm_state_dict[f"bias_ih_l{lyr}"]      # [4*hidden_size]
            bias_hh = lstm_state_dict[f"bias_hh_l{lyr}"]      # [4*hidden_size]
    
            # Split weights and biases into gate-specific parts and convert to CasADi
            weight_ih_split = torch.split(weight_ih, self.hidden_size, dim=0)
            weight_hh_split = torch.split(weight_hh, self.hidden_size, dim=0)
            bias_ih_split = torch.split(bias_ih, self.hidden_size, dim=0)
            bias_hh_split = torch.split(bias_hh, self.hidden_size, dim=0)

            for i, i_idx in enumerate(i_indices):
                self.W[i_idx + f'_{lyr}'] = cas.DM(weight_ih_split[i].detach().numpy())
                self.b[i_idx + f'_{lyr}'] = cas.DM(bias_ih_split[i].detach().numpy())

            for i, h_idx in enumerate(h_indices):
                self.W[h_idx + f'_{lyr}'] = cas.DM(weight_hh_split[i].detach().numpy())
                self.b[h_idx + f'_{lyr}'] = cas.DM(bias_hh_split[i].detach().numpy())

        # Extract FC weights and biases
        fc_state_dict = model.fc.state_dict()
        self.fc_weights = cas.DM(fc_state_dict["weight"].detach().numpy())
        self.fc_biases = cas.DM(fc_state_dict["bias"].detach().numpy())

    def forward(self, x):
        h0 = cas.DM.zeros(self.hidden_size, self.num_layers)
        c0 = cas.DM.zeros(self.hidden_size, self.num_layers)

        # Process input through LSTM
        out, _ = self.lstm(x, h0, c0)

        # Process last LSTM output through FC layer
        out = self.fc(out[-1])
        return out
    
    def lstm(self, x0, h0, c0):
        """
        Custom multi-layer LSTM implementation with proper handling of input and weight dimensions.

        Args:
            x0: Input tensor of shape (sequence_length, input_size).
            h0: Initial hidden state of shape (1, hidden_size).
            c0: Initial cell state of shape (1, hidden_size).

        Returns:
            out: Output tensor of shape (sequence_length, hidden_size).
            (h_t, c_t): Final hidden and cell states, both of shape (1, hidden_size).

        References:
            https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            https://arxiv.org/pdf/1503.04069
            https://arxiv.org/pdf/1402.1128
        """
        for lyr in range(self.num_layers):
            outputs = []
            h_prev, c_prev = h0[:,lyr], c0[:,lyr]
            for t in range(x0.size2()):
                if lyr == 0:
                    x_t = x0[:,t]
                else:
                    x_t = out[t]

                # Compute gate activations
                i_t = sigmoid(self.W[f'ii_{lyr}'] @ x_t + self.b[f'ii_{lyr}'] + self.W[f'hi_{lyr}'] @ h_prev + self.b[f'hi_{lyr}'])
                f_t = sigmoid(self.W[f'if_{lyr}'] @ x_t + self.b[f'if_{lyr}'] + self.W[f'hf_{lyr}'] @ h_prev + self.b[f'hf_{lyr}'])
                g_t = cas.tanh(self.W[f'ig_{lyr}'] @ x_t + self.b[f'ig_{lyr}'] + self.W[f'hg_{lyr}'] @ h_prev + self.b[f'hg_{lyr}'])
                o_t = sigmoid(self.W[f'io_{lyr}'] @ x_t + self.b[f'io_{lyr}'] + self.W[f'ho_{lyr}'] @ h_prev + self.b[f'ho_{lyr}'])

                # Compute cell and hidden states
                c_t = f_t * c_prev + i_t * g_t
                h_t = o_t * cas.tanh(c_t)

                h_prev, c_prev = h_t, c_t

                outputs.append(h_t)
            out = outputs

        # out should have shape sequence_length x hidden_size
        return out, (h_t, c_t)
    
    def fc(self, x):
        return self.fc_weights @ x + self.fc_biases


if __name__ == '__main__':
    # Hyperparameters
    sequence_length = 3    # Number of past timesteps to look at for each prediction
    hidden_size = 32       # Number of features in the hidden state of the LSTM
    num_layers = 4         # Number of LSTM layers

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(num_layers, x.size(0), hidden_size)
            c0 = torch.zeros(num_layers, x.size(0), hidden_size)
            
            out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, sequence_length, hidden_size)
            out = self.fc(out[:, -1, :])  # Select the last time step for the output
            return out
    
    # Initialize model
    input_size = 9
    output_size = 2
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    # print(f'Loading model state dict from {model_path}')
    # model.load_state_dict("/path/to/model.pth")
    # model.eval()

    model_cas = LSTMCasADi(model)
    opti = cas.Opti()
    x = opti.variable(9, sequence_length)
    tic = time()
    y = model_cas.forward(x)
    toc = time()
    print(f"Time for forward call: {(toc-tic)*1e3:.2f} ms")


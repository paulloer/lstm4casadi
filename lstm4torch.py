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
from time import time
torch.manual_seed(42)

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

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
                self.W[i_idx + f'_{lyr}'] = weight_ih_split[i]
                self.b[i_idx + f'_{lyr}'] = bias_ih_split[i]

            for i, h_idx in enumerate(h_indices):
                self.W[h_idx + f'_{lyr}'] = weight_hh_split[i]
                self.b[h_idx + f'_{lyr}'] = bias_hh_split[i]

        # Extract FC weights and biases
        fc_state_dict = model.fc.state_dict()
        self.fc_weights = fc_state_dict["weight"]
        self.fc_biases = fc_state_dict["bias"]

    def forward(self, xud):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)

        # Process input through LSTM
        out, _ = self.lstm(xud, h0, c0)

        # Process LSTM output through FC layer
        out = self.fc(out[-1,:])
        return out

    def lstm(self, x0, h0, c0):
        """
        Custom single-layer LSTM implementation with proper handling of input and weight dimensions.

        Args:
            x0: Input tensor of shape (sequence_length, input_size).
            h0: Initial hidden state of shape (1, hidden_size).
            c0: Initial cell state of shape (1, hidden_size).

        Returns:
            out: Output tensor of shape (sequence_length, hidden_size).
            (h_t, c_t): Final hidden and cell states, both of shape (1, hidden_size).

        # from PyTorch documentation:
        # i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
        # f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
        # g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
        # o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
        # c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
        # h_t = o_t \odot \tanh(c_t) \\
        """
    
        
        for lyr in range(self.num_layers):
            outputs = []
            h_prev, c_prev = h0[lyr], c0[lyr]
            for t in range(x0.size(0)):
                if lyr == 0:
                    x_t = x0[t]
                else:
                    x_t = out[t]

                # Compute gate activations
                i_t = sigmoid(x_t @ self.W[f'ii_{lyr}'].T + self.b[f'ii_{lyr}'] + h_prev @ self.W[f'hi_{lyr}'].T + self.b[f'hi_{lyr}'])
                f_t = sigmoid(x_t @ self.W[f'if_{lyr}'].T + self.b[f'if_{lyr}'] + h_prev @ self.W[f'hf_{lyr}'].T + self.b[f'hf_{lyr}'])
                g_t = torch.tanh(x_t @ self.W[f'ig_{lyr}'].T + self.b[f'ig_{lyr}'] + h_prev @ self.W[f'hg_{lyr}'].T + self.b[f'hg_{lyr}'])
                o_t = sigmoid(x_t @ self.W[f'io_{lyr}'].T + self.b[f'io_{lyr}'] + h_prev @ self.W[f'ho_{lyr}'].T + self.b[f'ho_{lyr}'])

                # Compute cell and hidden states
                c_t = f_t * c_prev + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

                h_prev, c_prev = h_t, c_t

                outputs.append(h_t)
            out = torch.stack(outputs, dim=0)

        # out should have shape sequence_length x hidden_size
        return out, (h_t, c_t)
    
    def fc(self, x):
        return self.fc_weights @ x + self.fc_biases


if __name__ == '__main__':
    # Hyperparameters
    sequence_length = 3    # Number of past timesteps to look at for each prediction
    hidden_size = 32      # Number of features in the hidden state of the LSTM
    num_layers = 4         # Number of LSTM layers
    learning_rate = 0.0001
    num_epochs = 30
    batch_size = 16
    weight_decay = 0.0001
    config_string = f'{sequence_length}sl_nn-{hidden_size}-{num_layers}_{learning_rate}lr_{num_epochs}e_{batch_size}bs_{weight_decay}wd_resampled_10min'
    model_path = f"./lstm_for_control/lstm_greenhouse_model_{config_string}.pth"
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    num_workers = 64 if device.type == "cuda" else 0

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, xud):
            # l4casadi only accepts vectors
            xud = xud.view(1, sequence_length, 9)
            
            h0 = torch.zeros(num_layers, xud.size(0), hidden_size).to(xud.device)
            c0 = torch.zeros(num_layers, xud.size(0), hidden_size).to(xud.device)
            
            out, _ = self.lstm(xud, (h0, c0))  # out: tensor of shape (batch_size, sequence_length, hidden_size)
            out = self.fc(out[:, -1, :])  # Select the last time step for the output
            return out
    
    # Initialize model
    input_size = 9
    output_size = 2
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    print(f'Loading model state dict from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device.type), weights_only=True))
    model.eval()

    model_cas = LSTMCasADi(model)

    batch_size = 1
    x = torch.rand(sequence_length,input_size)

    y1 = model(x)
    y2 = model_cas.forward(x)


    print(y1)
    print(y2)



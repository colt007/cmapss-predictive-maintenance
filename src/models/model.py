import torch
import torch.nn as nn

class CMAPSSModel(nn.Module):
    def __init__(self,input_size:int,hidden_size:int=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
            )
        self.fc1 = nn.Linear(hidden_size,64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,1)
        
    def forward(self,x):
        lstm_out,(h_n,c_n) = self.lstm(x)
        last_hidden = lstm_out[:,-1,:]
        x = self.fc1(last_hidden)
        x= self.relu(x)
        x= self.fc2(x)
        return x
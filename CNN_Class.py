import torch
import torch.nn as nn

class CNN_Class(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_Class, self).__init__()
        
        # The input size is (batch_size, seq_len, num_features)
 

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=10, kernel_size=3, stride=1, padding=1),  # Match output channels to input features
            nn.ReLU()
        )
        
        # # Calculate the input size for LSTM
        # lstm_input_size = 128  # 128 channels and seq_len remains same due to stride=1

        # self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, 1)
        
        
       
            
        
    

    def forward(self, x):
        # Convert the data to torch tensors
        x = torch.from_numpy(x).float()
        
        
      
        
        # Permute to match the input shape expected by Conv1d
        
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        
        out = self.cnn(x)
        
        # Reshape output of CNN to match the input shape expected by LSTM
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, num_features)
        
        # # # Flatten the features to match LSTM input
        # out = out.contiguous().view(out.size(0), out.size(1), -1)
        
        # # LSTM
        # out, _ = self.lstm(out)
        
        # batch_size, seq_length, hidden_size = out.shape
        # out = out.reshape(-1, hidden_size)
        
        # # Apply the fully connected layer
        # out = self.fc(out)
        
        # # Reshape back to (batch_size, seq_length, output_size)
        # out = out.view(batch_size, seq_length, -1)
        
        # FC
        # out = self.fc(out[:, -1, :])
        
        return out

# # Example usage
# input_size = (32, 3, 10)  # (batch_size, seq_len, num_features)
# hidden_size = 128
# num_layers = 2
# num_classes = 1  # Adjust according to your problem

# model = CNN_Class(input_size, hidden_size, num_layers, num_classes)

# # Dummy input for testing
# x = torch.randn(32, 3, 10)
# output = model(x)
# print(output.shape)

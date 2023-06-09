from torch import nn
import torch
from torchsummary import summary
import torch.nn.functional as F


# class CNNNetwork(nn.Module):
#     def __init__(self, n_input=1, n_output=2, stride=16, out_channel=32):
#         super(CNNNetwork,self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=out_channel, kernel_size=5, stride=2)
#         # self.bn1 = nn.BatchNorm1d(n_channel)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=5)
#         # self.bn2 = nn.BatchNorm1d(n_channel)change this   
#         self.pool2 = nn.MaxPool1d(4) # TODO: change this in future
#         self.fc1 = nn.Linear(out_channel, n_output)
#         # self.fc2 = nn.Linear(n_channel, n_output)

#     def forward(self, x):
#         # print('forward')
#         # print(x.shape)
#         x = self.conv1(x)
#         x = F.relu((x))
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu((x))
#         x = self.pool2(x)
#         # print(x.shape)
#         x = F.avg_pool1d(x, x.shape[-1])
#         x = x.permute(0, 2, 1)
#         x = self.fc1(x)
#         # x = self.fc2(x)
#         # print('linear size',x.shape)
#         return F.log_softmax(x, dim=-1)

# Simple CNN
class AudioEncoder(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=8,kernel_size=5,stride=2,padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16*3998, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # print('original', x.shape)
        x = F.relu((self.conv1(x)))
        # print('conv1', x.shape)
        x = self.pool(x)
        # print('ppoling 1',x.shape)
        x = F.relu(self.conv2(x))
        # print('conv2',x.shape)
        x = self.pool(x)
        # print('pooling 2',x.shape)
        x = x.reshape(x.shape[0], -1)
        # print('reshaping',x.shape)
        x = self.fc1(x)
        # print('fully connected',x.shape)
        return x
    
# batch_size = 256
# in_channels = 1
# out_channels = 1
# sequence_length = 1000

if __name__ == "__main__":
    model = AudioEncoder(n_input=1, n_output=2)
    device=('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(model)
    # cnn = CNNNetwork(in_channels, out_channels)
    # summary(cnn.cuda(), (1, 5040))


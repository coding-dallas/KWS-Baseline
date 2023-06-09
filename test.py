import torch
import torch.nn.functional as F  
import torchaudio
from audio_dataloader import AudioData
import torchvision.transforms as transforms 
from torch import optim 
from cnn import AudioEncoder
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm 

# Train Network
def train(model, criterion, train_dataloader, num_epochs):
    temp={}
    loss_temp={}
    
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            data = data.view(data.shape[0], data.shape[1], -1)
            # print(data.shape)
            # forward
            scores = model(data)
            _, predictions = scores.max(1)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent or adam step
            optimizer.step()
            
        if epoch%5==0:
            temp[epoch]=predictions
            loss_temp[epoch]=loss
        
        print(f"Accuracy on training set: {check_accuracy(train_dataloader, model)*100:.2f}")
        print(f"Accuracy on test set: {check_accuracy(test_dataloader, model)*100:.2f}")

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # print(x.shape)
            x = x.view(x.shape[0], x.shape[1], -1)
            # print(x.shape)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

if __name__ == '__main__':
    sample_rate=16000
    num_samples=31840
    win_length =int(0.025 * sample_rate)
    hop_length =int(0.01 * win_length)

    #path
    train_metadata = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/metadata/train_1word.csv'
    test_metadata = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/metadata/test_1word.csv'
    data_dir = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/database/LibriPhrase_diffspk_all'
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        # win_length=win_length,
        hop_length=160,
        n_mels=80
    )
    
    #loading_data from directory
    train_data=AudioData(
            train_metadata, 
            data_dir, 
            mel_spectrogram, 
            sample_rate, 
            num_samples, 
            device
        )
    test_data=AudioData(
            test_metadata, 
            data_dir, 
            mel_spectrogram, 
            sample_rate, 
            num_samples, 
            device
        )
    
    #hyperparameters
    num_epochs = 10
    batch_size = 32
    in_channels = 1
    num_classes = 2
    learning_rate = 0.001
    # log_interval = 20
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size,drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,drop_last=True, shuffle=True) 
    
    #chekcing the data shape
    for i in train_dataloader:
        print(i[0].shape)
        print(i[1].shape)
        break
    model = AudioEncoder(in_channels=in_channels, num_classes=num_classes).to(device)
    print(model.parameters())

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    train(model, criterion, train_dataloader, num_epochs)
        
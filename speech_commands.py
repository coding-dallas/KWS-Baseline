import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from audio_dataloader import AudioData
from cnn import CNNNetwork
from tqdm import tqdm
import matplotlib.pyplot as plt



def create_data_loader(data, batch_size): #train_data shape is torch.Size([1, 80, 126])
    dataloader = DataLoader(data, batch_size=batch_size,drop_last=True, shuffle=True)
    return dataloader

def train(model, epoch, log_interval, train_loader, loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(target)
        data = data.view(data.shape[0], data.shape[1], -1)
        
        data = data.to(device)
        target = target.to(device)
        
        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output = model(data)
        # print(output.shape, target.shape)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = loss_fn(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
        
def number_of_correct(pred, target, epoch):
    # if epoch%10==0:
    #     print(pred)
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    # tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    print(tensor)
    return tensor


def test(model, epoch, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        # print(target)
        data = data.view(data.shape[0], data.shape[1], -1)
            
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target, epoch)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    
    BATCH_SIZE = 128
    log_interval = 20
    n_epoch = 50
    in_channels = 1
    out_channels = 2
    LEARNING_RATE = 0.001

    train_metadata = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/metadata/train_1word.csv'
    test_metadata = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/metadata/test_1word.csv'
    data_dir = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/database/LibriPhrase_diffspk_all'
    
    SAMPLE_RATE=16000
    NUM_OF_SAMPLES=31840

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )

    train_dataload=AudioData(
        train_metadata, 
        data_dir, 
        mel_spectrogram, 
        SAMPLE_RATE, 
        NUM_OF_SAMPLES, 
        device
    )
    
    test_dataload=AudioData(
        test_metadata, 
        data_dir, 
        mel_spectrogram, 
        SAMPLE_RATE, 
        NUM_OF_SAMPLES, 
        device
    )
    
    
    train_dataloader = create_data_loader(train_dataload, BATCH_SIZE)
    test_dataloader = create_data_loader(test_dataload, BATCH_SIZE)
    
    print(len(train_dataload), len(test_dataload))
    for i in train_dataloader:
        print(i[0].shape)
        break
    
    model = CNNNetwork(in_channels, out_channels).to(device)
    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    pbar_update = 1 / (len(train_dataloader) + len(test_dataloader))
    losses = []
    
    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            # train(model, epoch, log_interval,train_dataloader, loss_fn)
            test(model, epoch, test_dataloader)
            # scheduler.step()
    
    torch.save(model.state_dict(), "./models/conv1D.pth")
    print("Trained feed forward net saved at conv1D.pth")        
    
    # # signal,label = train_dataload[-1]

    # # print(f"Expected: {label}. Predicted: {predict(signal.view(signal.shape[0],signal.shape[1],-1))}.")
    
    # plt.plot(losses);
    # plt.title("training loss");
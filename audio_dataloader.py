import   os
import torchaudio
import pandas as pd
import torch
from torch.utils.data import Dataset

class AudioData(Dataset):
    
    def __init__(self,metadata, data_dir, transformation, SAMPLE_RATE, NUM_OF_SAMPLES, device):
        self.metadata = pd.read_csv(metadata)
        self.audio_dir = data_dir
        self.device=device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = SAMPLE_RATE
        self.num_of_samples = NUM_OF_SAMPLES
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        anchor_audio_path = self._get_anchor_audio_path(index)
        comparison_audio_path = self._get_comparisons_audio_path(index)
        
        # print(audio_sample_path)
        label = self._get_audio_sample_label(index)
        
        anchor_signal, a_sr = torchaudio.load(anchor_audio_path)
        comparison_signal, c_sr = torchaudio.load(comparison_audio_path)
        
        anchor_signal= anchor_signal.to(self.device)
        comparison_signal= comparison_signal.to(self.device)
        
        anchor_signal = self._resample_if_necessary(anchor_signal, a_sr)
        comparison_signal = self._resample_if_necessary(comparison_signal, c_sr)
        
        anchor_signal = self._mix_down_if_necessary(anchor_signal)
        comparison_signal = self._mix_down_if_necessary(comparison_signal)
        
        anchor_signal = self.right_pad_if_necessary(anchor_signal)
        comparison_signal = self.right_pad_if_necessary(comparison_signal)
        
        anchor_signal = self.transformation(anchor_signal)
        comparison_signal = self.transformation(comparison_signal)
        
        concate_signal= torch.cat((anchor_signal,comparison_signal), 2)
        
        return concate_signal, label
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_of_samples:
            num_missing_samples = self.num_of_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _get_anchor_audio_path(self, index):
        sub_path = f"{self.metadata.iloc[index, 0]}" 
        path = os.path.join(self.audio_dir, sub_path)
        return path
    
    def _get_comparisons_audio_path(self, index):
        sub_path = f"{self.metadata.iloc[index, 4]}" 
        path = os.path.join(self.audio_dir, sub_path)
        return path

    def _get_audio_sample_label(self, index):
        return self.metadata.iloc[index, 9]  
    
        
if __name__ == '__main__':
    metadata = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/metadata/testset_librispeech_other_train_100h_short_phrase_1word.csv'
    data_dir = '/home/kesav/Documents/kesav/research/code_files/LibriPhrase/database/LibriPhrase_diffspk_all'
    SAMPLE_RATE=16000
    NUM_OF_SAMPLES=31840
    
    device=('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for device')
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=80
    )
    
    dataload=AudioData(metadata, data_dir, mel_spectrogram, SAMPLE_RATE, NUM_OF_SAMPLES, device)
    
    print(f"There are {len(dataload)} samples in the dataset.")
    signal, label = dataload[16]
    print(dataload.signal)
from torch.utils.data import Dataset
from pathlib import Path
from torch import Tensor
import librosa
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
import numpy as np
import torchaudio

class CustomDataset(Dataset):
    """
    description: custom dataset class for reading data from a directory (assignmet data provied)
    Note: lables are assigned as 0 for real and 1 for fake
    args:
        dir: str - path to the directory containing the data
    """
    def __init__(self,dir:str, config) -> None:
        super().__init__()
        self.dir = Path(dir)
        self.config = config
        self.cut=64600 # take ~4 sec audio (64600 samples)
        types = ('*.wav', '*.mp3')
        # Get a list of files matching the extensions
        real_audio = []
        for files in types:
            real_audio.extend(list(self.dir.glob(f"Real/{files}")))
            
        fake_audio = []
        for files in types:
            fake_audio.extend(list(self.dir.glob(f"Fake/{files}")))

        real_audio = sorted(map(str,real_audio))
        # print(f"[INFO] lenth of real audio: {len(real_audio)}") => 180
        fake_audio = sorted(map(str,fake_audio))
        # print(f"[INFO] lenth of fake audio: {len(fake_audio)}") => 120
        self.complete_data = [(audio,0) for audio in real_audio] + [(audio,1) for audio in fake_audio]
        
    def __len__(self) -> int:
        """
        description: returns the length of the dataset (real + fake)
        """
        return len(self.complete_data)
    
    def __getitem__(self, index) -> tuple:
        """
        description: returns the audio file and its label
        args:
            index: int - index of the audio file
        returns:
            tuple: audio file encoded as per config and its label
        """
        X,fs = librosa.load(self.complete_data[index][0], sr=16000)
        Y =  process_Rawboost_feature(X,fs,self.config,self.config.algo)
        X_pad= pad(Y,self.cut)
        x_inp= Tensor(X_pad)
        target = self.complete_data[index][1]
            
        return x_inp, target
    

class FOR2SsecDataset(CustomDataset):
    """
    description: custom dataset class for reading data from a directory (for-2seconds dataset provied)
    args:
        dir: str - path to the directory containing the data
        config: Config - configuration object
        split: str - training, testing or validation
    """
    def __init__(self,dir:str, config, split:str="training") -> None:
        super().__init__(dir, config)
        assert split in ["training", "testing", "validation"], "split should be either training, testing or validation"
        self.cut=32000
        self.config = config
        self.dir = Path(dir)
        types = ('*.wav', '*.mp3')
        # Get a list of files matching the extensions
        real_audio = []
        for files in types:
            real_audio.extend(list(self.dir.glob(f"{split}/real/{files}")))
        
        fake_audio = []
        for files in types:
            fake_audio.extend(list(self.dir.glob(f"{split}/fake/{files}")))
        
        real_audio = sorted(map(str,real_audio))
        # print(f"[INFO] lenth of real audio: {len(real_audio)}")
        fake_audio = sorted(map(str,fake_audio))
        # print(f"[INFO] lenth of fake audio: {len(fake_audio)}")
        self.complete_data = [(audio,0) for audio in real_audio] + [(audio,1) for audio in fake_audio]
        
    
        
    
##--------------Pad Function---------------------------##
# code take from SP/A3/SSL_Anti-spoofing/data_utils_SSL.py
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	


##--------------RawBoost data augmentation algorithms---------------------------##
# code take from SP/A3/SSL_Anti-spoofing/data_utils_SSL.py
def process_Rawboost_feature(feature,sr,config,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,config.N_f,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,config.minCoeff,config.maxCoeff,config.minG,config.maxG,config.minBiasLinNonLin,config.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, config.P, config.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,config.SNRmin,config.SNRmax,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,config.minCoeff,config.maxCoeff,config.minG,config.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,config.N_f,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,
                 config.minCoeff,config.maxCoeff,config.minG,config.maxG,config.minBiasLinNonLin,config.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, config.P, config.g_sd)  
        feature=SSI_additive_noise(feature,config.SNRmin,config.SNRmax,config.nBands,config.minF,
                config.maxF,config.minBW,config.maxBW,config.minCoeff,config.maxCoeff,config.minG,config.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,config.N_f,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,
                 config.minCoeff,config.maxCoeff,config.minG,config.maxG,config.minBiasLinNonLin,config.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, config.P, config.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,config.N_f,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,
                 config.minCoeff,config.maxCoeff,config.minG,config.maxG,config.minBiasLinNonLin,config.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,config.SNRmin,config.SNRmax,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,config.minCoeff,config.maxCoeff,config.minG,config.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, config.P, config.g_sd)
        feature=SSI_additive_noise(feature,config.SNRmin,config.SNRmax,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,config.minCoeff,config.maxCoeff,config.minG,config.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,config.N_f,config.nBands,config.minF,config.maxF,config.minBW,config.maxBW,
                 config.minCoeff,config.maxCoeff,config.minG,config.maxG,config.minBiasLinNonLin,config.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, config.P, config.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature



# test the custom dataset class
# if __name__ == "__main__":
#     import argparse
#     from tqdm.auto import tqdm
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--N_f', type=int, default=1)
#     parser.add_argument('--nBands', type=int, default=1)
#     parser.add_argument('--minF', type=int, default=1)
#     parser.add_argument('--maxF', type=int, default=1)
#     parser.add_argument('--minBW', type=int, default=1)
#     parser.add_argument('--maxBW', type=int, default=1)
#     parser.add_argument('--minCoeff', type=int, default=1)
#     parser.add_argument('--maxCoeff', type=int, default=1)
#     parser.add_argument('--minG', type=int, default=1)
#     parser.add_argument('--maxG', type=int, default=1)
#     parser.add_argument('--minBiasLinNonLin', type=int, default=1)
#     parser.add_argument('--maxBiasLinNonLin', type=int, default=1)
#     parser.add_argument('--P', type=int, default=1)
#     parser.add_argument('--g_sd', type=int, default=1)
#     parser.add_argument('--SNRmin', type=int, default=1)
#     parser.add_argument('--SNRmax', type=int, default=1)
#     parser.add_argument('--algo', type=int, default=3)
#     config = parser.parse_args()
#     dataset = CustomDataset('data/Dataset_Speech_Assignment',config)
#     for2sec_dataset = FOR2SsecDataset('data/for-2seconds',config)
#     # print(dataset[0])
#     for i in tqdm(dataset):
#         pass
    
#     for i in tqdm(for2sec_dataset):
#         pass
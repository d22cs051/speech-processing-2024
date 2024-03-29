import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import torchmetrics
import wandb

# wandb.init(project="minor-audio-source-separation")


# defining dataset class
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_files = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        src1_name = self.data_files.iloc[idx, 0]
        src2_name = self.data_files.iloc[idx, 1]
        mix_name = self.data_files.iloc[idx, 2]

        src1, _ = torchaudio.load(self.data_dir / "source1" / src1_name)
        src2, _ = torchaudio.load(self.data_dir / "source2" / src2_name)
        mix, _ = torchaudio.load(self.data_dir / "mixture"/ mix_name)

        if self.transform:
            src1 = self.transform(src1)
            src2 = self.transform(src2)
            mix = self.transform(mix)

        return src1, src2, mix

# Test the dataset class
data_root = Path("data")
audio_dataset = AudioDataset(csv_file=data_root / "info.csv",data_dir=data_root)

# for i in tqdm(audio_dataset):
#     pass

# spliting the dataset into train and test
train_audio_dataset, test_audio_dataset = torch.utils.data.random_split(audio_dataset, [0.9, 0.1])
print(f"Train dataset size: {len(train_audio_dataset)}")
print(f"Test dataset size: {len(test_audio_dataset)}")

# defining the dataloader
BATCH_SIZE = 4
train_dataloader = DataLoader(train_audio_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_audio_dataset, batch_size=BATCH_SIZE, shuffle=False)

# defining the model LSTM for source separation s1 and s2 from mixture x, input shape (batch, 1, 20000)
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # input shape (batch, 1, 20000)
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(20000, 128, 8, batch_first=True)
        self.fc1 = nn.Linear(128, 20000)
        self.fc2 = nn.Linear(128, 20000)

    def forward(self, x):
        # x is the mixture of s1 and s2
        # print(f"[LOG] x.shape, {x.shape}")
        x = self.flatten(x)
        # print(f"[LOG] x.shape, {x.shape}")
        x, _ = self.lstm(x)
        # print(f"[LOG] x.shape, {x.shape}")
        s1 = self.fc1(x)
        s2 = self.fc2(x)
        # print(f"[LOG] s1.shape, {s1.shape}, s2.shape, {s2.shape}")
        return s1, s2

# defining the deice
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# defining the model
model = LSTM()
model.to(DEVICE)

# defining the loss function
criterion = nn.L1Loss()
si_snr = torchmetrics.audio.ScaleInvariantSignalNoiseRatio()
si_snr.to(DEVICE)

# defining the optimizer
LR = 3e-4
optimizer = optim.Adam(model.parameters(), lr=LR)


# training the model
EPOCHS = 10

# wandb.run.name = f"Q1-LSTM-Source-Separation-lr-{LR}-epochs-{EPOCHS}-batch_size-{BATCH_SIZE}"
for epoch in tqdm(range(EPOCHS)):
    model.train()
    running_loss = 0.0
    running_si_snr = 0.0
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        src1, src2, mix = data
        src1, src2, mix = src1.squeeze().to(DEVICE), src2.squeeze().to(DEVICE), mix.to(DEVICE)

        # print(mix.shape)
        s1, s2 = model(mix)

        optimizer.zero_grad()
        loss = criterion(s1, src1) + criterion(s2, src2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_si_snr += si_snr(s1, src1) + si_snr(s2, src2)
        # batch logs
        # wandb.log({"Batch Train Loss": loss.item(), "Batch Train SI-SNR": si_snr(s1, src1) + si_snr(s2, src2)})
    # print(f"Epoch {epoch+1},Train Loss: {running_loss/len(train_dataloader)},Train SI-SNR: {running_si_snr/len(train_dataloader)}")
    # wandb.log({"AVG. Train Loss": running_loss/len(train_dataloader), "AVG. Train SI-SNR": running_si_snr/len(train_dataloader), "Epoch": epoch+1})
    
    model.eval()
    running_loss = 0.0
    running_si_snr = 0.0
    with torch.inference_mode():
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            src1, src2, mix = data
            src1, src2, mix = src1.squeeze().to(DEVICE), src2.squeeze().to(DEVICE), mix.to(DEVICE)

            s1, s2 = model(mix)
            # print(s1.shape, src1.shape, s2.shape, src2.shape)
            # save the output audio
            data_root = Path("data")
            (data_root/"output"/"input1").mkdir(exist_ok=True, parents=True)
            (data_root/"output"/"input2").mkdir(exist_ok=True, parents=True)
            (data_root/"output"/"source1").mkdir(exist_ok=True, parents=True)
            (data_root/"output"/"source2").mkdir(exist_ok=True, parents=True)
            for _ in range(int(src1.shape[0])):
                torchaudio.save(data_root / "output" / "input1" / f"batch_{_}_input1_{i}.wav", src1[_].unsqueeze(0).cpu(), 16000)
                torchaudio.save(data_root / "output" / "input2" / f"batch_{_}_input2_{i}.wav", src2[_].unsqueeze(0).cpu(), 16000)
                torchaudio.save(data_root / "output" / "source1" / f"batch_{_}_output1_{i}.wav", s1[_].unsqueeze(0).cpu(), 16000)
                torchaudio.save(data_root / "output" / "source2" / f"batch_{_}_output2_{i}.wav", s2[_].unsqueeze(0).cpu(), 16000)
            
            
            test_loss = criterion(s1, src1) + criterion(s2, src2)
            
            running_loss += test_loss.item()
            running_si_snr += si_snr(s1, src1) + si_snr(s2, src2)
            # batch logs
            # wandb.log({"Batch Test Loss": test_loss.item(), "Batch Test SI-SNR": si_snr(s1, src1) + si_snr(s2, src2)})
    # print(f"Test Loss: {running_loss/len(test_dataloader)}, Test SI-SNR: {running_si_snr/len(test_dataloader)}")
    # wandb.log({"AVG. Test Loss": running_loss/len(test_dataloader), "AVG. Test SI-SNR": running_si_snr/len(test_dataloader)})

# saving the model in ckpt
torch.save(model.state_dict(), "model.ckpt")

print("Model Trained and Saved Successfully")
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
import json
from torch.nn import functional as F

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# defining dataset class
class HyperValleyDataset(torch.utils.data.Dataset):
    def __init__(self, json_files_dir,agent_audios_dir, caller_audios_dir, transform=None,tokenizer=None):
        self.json_files = sorted(list(json_files_dir.glob("*.json")))
        self.agent_audios_dir = agent_audios_dir
        self.caller_audios_dir = caller_audios_dir
        self.transform = transform
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.tokenizer = tokenizer

        
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        json_file = self.json_files[idx]
        with open(json_file, "r") as f:
            data = json.load(f)
        # print(json_file)
        # print(data)
        audio_file_id = json_file.stem
        # print(audio_file_id)
        agent_audio_file = self.agent_audios_dir / f"{audio_file_id}.wav"
        # print(agent_audio_file)
        agent_audio, _ = torchaudio.load(agent_audio_file)
        caller_audio_file = self.caller_audios_dir / f"{audio_file_id}.wav"
        caller_audio, _ = torchaudio.load(caller_audio_file)
        avg_emotion_dict = {"emotion": {
            "neutral": 0.0,
            "negative": 0.0,
            "positive": 0.0
                    }
                       }
        transcript = ""
        for i in data:
            if i["speaker_role"] == "agent":
                transcript += i["human_transcript"] + " "

            elif i["speaker_role"] == "caller":
                transcript += i["human_transcript"]+ " "
            avg_emotion_dict["emotion"]["neutral"] += i["emotion"]["neutral"]
            avg_emotion_dict["emotion"]["negative"] += i["emotion"]["negative"]
            avg_emotion_dict["emotion"]["positive"] += i["emotion"]["positive"]
        avg_emotion_dict["emotion"]["neutral"] /= len(data)
        avg_emotion_dict["emotion"]["negative"] /= len(data)
        avg_emotion_dict["emotion"]["positive"] /= len(data)
        # print(avg_emotion_dict)
        # making the audio files of same length
        if agent_audio.shape[1] > caller_audio.shape[1]:
            agent_audio = agent_audio[:, :caller_audio.shape[1]]
        else:
            caller_audio = caller_audio[:, :agent_audio.shape[1]]
        combined_audio = torch.concatenate((agent_audio, caller_audio), dim=0)
        # print(transcript)
        if self.transform:
            combined_audio = self.transform(combined_audio)
        
        # torchaudio.save("combined_audio.wav", combined_audio, _)
        combined_audio = torchaudio.functional.resample(combined_audio, _ , self.bundle.sample_rate)
        mfcc_features = torchaudio.transforms.MFCC()(combined_audio)[:,:,:1000]
        print(f"[LOG] MFCC Features Shape: {mfcc_features.shape}")
        return mfcc_features, tokenizer.convert_tokens_to_ids(transcript), self._emotion_dict_to_tensor(avg_emotion_dict)
    
    def _emotion_dict_to_tensor(self, emotion_dict):
        emotion_tensor = torch.tensor([emotion_dict["emotion"]["neutral"], emotion_dict["emotion"]["negative"], emotion_dict["emotion"]["positive"]])
        return emotion_tensor
  
# dataset definition  
dataset = HyperValleyDataset(Path("data/HarperValleyDataset_transcript/HarperValley_transcript"), Path("data/agent/agent"), Path("data/caller/caller"), tokenizer=tokenizer)
# Test of the dataset
# for i in dataset:
#     print(i)
#     break
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8,0.2])
print(f"[INFO] Train Dataset Length: {len(train_dataset)}")
print(f"[INFO] Val Dataset Length: {len(val_dataset)}")


# defining dataloader
BATCH_SIZE = 16
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# defining model
class ASRModel(nn.Module):
    def __init__(self):
        super(ASRModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(64, 128, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(47232, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        return self.model(x)
    

model = ASRModel()
print(model)

# defining device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining loss function
loss_fn1 = nn.CTCLoss()
loss_fn2 = nn.CrossEntropyLoss()

# defining metrics
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=3)
accuracy_fn = accuracy_fn.to(device)

# defining optimizer
LR = 1e-3
optimizer = optim.Adam(model.parameters(), lr=LR)


# training the model
EPOCHS = 10
model.to(device)
for epoch in tqdm(range(EPOCHS),desc="Epochs"):
    model.train()
    loop = tqdm(train_dataloader, leave=True)
    running_acc = 0
    running_loss1 = 0
    running_loss2 = 0
    for mfcc_features, transcript, emotion in loop:
        mfcc_features, transcript, emotion = mfcc_features.to(device), transcript.to(device), emotion.to(device)
        optimizer.zero_grad()
        output = model(mfcc_features)
        # print(output.shape, transcript.shape)
        # print(output, transcript)
        loss1 = loss_fn1(output, transcript)
        loss2 = loss_fn2(output, emotion)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item())
        running_acc += accuracy_fn(output, emotion)
        running_loss1 += loss1.item()
        running_loss2 += loss2.item()
        
    print(f"Epoch: {epoch+1}/{EPOCHS}, Loss1: {running_loss1/len(train_dataloader)}, Loss2: {running_loss2/len(train_dataloader)}, Accuracy: {running_acc/len(train_dataloader)}")
    # validation
    model.eval()
    running_acc = 0
    running_loss1 = 0
    running_loss2 = 0
    with torch.inference_mode():
        val_loop = tqdm(test_dataloader, leave=True)
        for mfcc_features, transcript, emotion in val_loop:
            mfcc_features, transcript, emotion = mfcc_features.to(device), transcript.to(device), emotion.to(device)
            output = model(mfcc_features)
            loss1 = loss_fn1(output, transcript)
            loss2 = loss_fn2(output, emotion)
            loss = loss1 + loss2
            val_loop.set_description(f"Epoch: {epoch+1}/{EPOCHS}")
            val_loop.set_postfix(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item())
            running_acc += accuracy_fn(output, emotion)
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
        print(f"Validation, Loss1: {running_loss1/len(test_dataloader)}, Loss2: {running_loss2/len(test_dataloader)}, Accuracy: {running_acc/len(test_dataloader)}")
    
# saving the model
torch.save(model.state_dict(), "multi-task-model.pth")
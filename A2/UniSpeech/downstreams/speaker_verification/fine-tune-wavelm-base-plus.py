# Description: This file is used to create a dataset for speaker verification fine-tuning.

from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_file
import os

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torchaudio
from verification import init_model
import wandb
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


## EER calculation
def cal_eer(y, y_score):
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

## Dataset class for Speaker Verification fine-tuning
EFFECTS = [
# ["channels", "1"],
# ["rate", "16000"],
["gain", "-3.0"],
["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
]

class SpeakerVerifi_test(Dataset):
    def __init__(self, vad_config, file_path, meta_data):
        self.root = file_path
        self.meta_data = meta_data
        self.necessary_dict = self.processing()
        self.vad_c = vad_config 
        self.dataset = self.necessary_dict['pair_table'] 
        
    def processing(self):
        pair_table = []
        with open(self.meta_data, "r") as f:
            usage_list = f.readlines()
        for pair in usage_list:
            list_pair = pair.split()
            pair_1= os.path.join(self.root, list_pair[1].split("/")[-1])
            pair_2= os.path.join(self.root, list_pair[2].split("/")[-1])
            one_pair = [list_pair[0],pair_1,pair_2 ]
            pair_table.append(one_pair)
        # print(f"printing pair_table: {pair_table[:2]}") # NOTE: testing purpose only
        return {
            "spk_paths": None,
            "total_spk_num": None,
            "pair_table": pair_table
        }

    def __len__(self):
        return len(self.necessary_dict['pair_table'])

    def __getitem__(self, idx):
        y_label, x1_path, x2_path = self.dataset[idx]
        def path2name(path):
            return path#Path("-".join((Path(path).parts)[-3:])).stem

        x1_name = path2name(x1_path)
        x2_name = path2name(x2_path)

        wav1, _ = apply_effects_file(x1_path, EFFECTS)
        wav2, _ = apply_effects_file(x2_path, EFFECTS)

        wav1 = wav1.squeeze(0)
        wav2 = wav2.squeeze(0)

        
        return wav1.numpy(), wav2.numpy(), x1_name, x2_name, int(y_label[0])

    def collate_fn(self, data_sample):
        wavs1, wavs2, x1_names, x2_names, ylabels = zip(*data_sample)
        all_wavs = wavs1 + wavs2
        all_names = x1_names + x2_names
        return all_wavs, all_names, ylabels
    
    
# Define the dataset
hindi_dataset = SpeakerVerifi_test(file_path="/DATA1/bikash_dutta/CS/SP/A2/UniSpeech/data/kb_data_clean_m4a/hindi/valid/wav", meta_data="/DATA1/bikash_dutta/CS/SP/A2/UniSpeech/data/kb_data_clean_m4a/meta_data/hindi/valid_data.txt", vad_config=None)
hindi_dataset_test = SpeakerVerifi_test(file_path="/DATA1/bikash_dutta/CS/SP/A2/UniSpeech/data/kb_data_clean_m4a/hindi/test_known/wav", meta_data="/DATA1/bikash_dutta/CS/SP/A2/UniSpeech/data/kb_data_clean_m4a/meta_data/hindi/test_known_data.txt", vad_config=None)


## Model class for Speaker Verification fine-tuning
# Loading wavlm_base_plus model for fine tuning


# model = init_model("wavlm_base_plus")
# print(model.eval())

class WaveLMSpeakerVerifi(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = init_model("wavlm_base_plus")
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, auido1, audio2):
        audio1_emb = self.feature_extractor(auido1)
        audio2_emb = self.feature_extractor(audio2)
        similarity = self.cosine_sim(audio1_emb, audio2_emb)
        similarity = (similarity + 1) / 2 # converting (-1,1) -> (0,1)
        return similarity



# Define the fine-tuning parameters
learning_rate = 1e-3
num_epochs = 15
batch_size = 32

# collate function to make the batch waves of same length
def collate_fn(data_sample):
    wavs1, wavs2, x1_names, x2_names, ylabels = zip(*data_sample)
    min_len = min([wav.shape[0] for wav in wavs1+wavs2])
    wavs1 = torch.Tensor([wav[:min_len] for wav in wavs1])
    wavs2 = torch.Tensor([wav[:min_len] for wav in wavs2])
    ylabels = torch.Tensor(ylabels)
    
    return wavs1, wavs2, x1_names, x2_names, ylabels
    


#  Define the fine-tuning dataloader
fine_tuning_dataloader = DataLoader(hindi_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
fine_tuning_dataloader_test = DataLoader(hindi_dataset_test, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

# Load the pre-trained model
pretrained_model = WaveLMSpeakerVerifi()


# Step 6: Define the loss function
loss_function = nn.BCEWithLogitsLoss()

# Step 7: Train the model
optimizer = optim.Adam(pretrained_model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)

pretrained_model.train()
print("Fine-tuning the model")



# Initialize Wandb logger
wandb.init(project="speaker_verification")

# naming experiment
wandb.run.name = "fine-tuning-wavlm-base-plus"

## Training loop
for epoch in tqdm(range(num_epochs), desc="Epoch Number:"):
    running_loss = 0
    running_eer = 0
    # Training Step
    pretrained_model.train()
    for batch_idx, batch in tqdm(enumerate(fine_tuning_dataloader), desc="Batch Number:",total=len(fine_tuning_dataloader)):
        wav1, wav2, x1_names, x2_names, ylabels = batch
        wav1 = wav1.to(device)
        wav2 = wav2.to(device)
        ylabels = ylabels.to(device)
        optimizer.zero_grad()
        output = pretrained_model(wav1, wav2)
        # print(output,output.dtype)
        # print(ylabels, ylabels.dtype)
        loss = loss_function(output, ylabels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_eer += cal_eer(ylabels.cpu().numpy(), output.cpu().detach().numpy())[0]
        
        # logging the loss and EER batchwise
        wandb.log({"train batch_idx":batch_idx, "Train Batch Loss": loss.item(), "Train Batch EER": cal_eer(ylabels.cpu().numpy(), output.cpu().detach().numpy())[0]})
        
    # Calculate average loss per epoch
    avg_loss = running_loss / len(fine_tuning_dataloader)
    avg_eer = running_eer / len(fine_tuning_dataloader)

    
    # Print and log EER and loss
    print(f"Epoch: {epoch+1}, Train Loss: {avg_loss}, Train EER: {avg_eer}")
    wandb.log({"Epoch": epoch+1, "AVG. Train Loss": avg_loss, "AVG. Train EER": avg_eer})
    
    # Testing Step
    pretrained_model.eval()
    test_running_loss = 0
    test_running_eer = 0
    with torch.inference_mode():
        for batch_idx, batch in tqdm(enumerate(fine_tuning_dataloader_test), desc="Batch Number:",total=len(fine_tuning_dataloader_test)):
            wav1, wav2, x1_names, x2_names, ylabels = batch
            wav1 = wav1.to(device)
            wav2 = wav2.to(device)
            ylabels = ylabels.to(device)
            output = pretrained_model(wav1, wav2)
            test_running_loss += loss_function(output, ylabels.float()).item()
            test_running_eer += cal_eer(ylabels.cpu().numpy(), output.cpu().detach().numpy())[0]
            
            # logging the EER batchwise
            wandb.log({"test batch_idx":batch_idx, "Test Batch EER": cal_eer(ylabels.cpu().numpy(), output.cpu().detach().numpy())[0]})
        
    # Calculate average loss per epoch
    avg_test_loss = test_running_loss / len(fine_tuning_dataloader_test)
    avg_test_eer = test_running_eer / len(fine_tuning_dataloader_test)
    
    # Print and log EER and loss
    print(f"Epoch: {epoch+1}, Test Loss: {avg_test_loss}, Test EER: {avg_test_eer}")
    wandb.log({"Epoch": epoch+1, "AVG. Test Loss": avg_test_loss, "AVG. Test EER": avg_test_eer})


# saving checkpoint in dictionary
checkpoint = {
    "model": pretrained_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch
}
# saving to file
torch.save(checkpoint, "fine-tuning-wavlm-base-plus-checkpoint.ckpt")

# saving the checkpoint to wandb
wandb.save("fine-tuning-wavlm-base-plus-checkpoint.ckpt")

print("Fine-tuning completed successfully!")
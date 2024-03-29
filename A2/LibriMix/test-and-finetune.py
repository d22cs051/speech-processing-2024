from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio, torchmetrics, torch
import pandas as pd
from tqdm.auto import tqdm
import wandb


wandb.init(project="speech-separation")
wandb.run.name = "fine-tuning-sepformer-wsj03mix"

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir='pretrained_models/sepformer-wsj03mix', run_opts={"device": device})

# print(model)

# Initialize SISNRi and SDRi metrics
sisnri_metric = torchmetrics.audio.ScaleInvariantSignalNoiseRatio()
sisnri_metric = sisnri_metric.to(device=device)
sdr_metric = torchmetrics.audio.ScaleInvariantSignalDistortionRatio()
sdr_metric = sdr_metric.to(device=device)

# Define dataset class
class Libri2Mix(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mixture_id, mix_path, s1_path, s2_path, noise_path, length = self.data.iloc[idx]
        mix, _ = torchaudio.load(mix_path)
        s1, _ = torchaudio.load(s1_path)
        s2, _ = torchaudio.load(s2_path)
        noise, _ = torchaudio.load(noise_path)
        return mix[:,:100000], s1[:,:100000], s2[:,:100000], noise[:,:100000], length

# collate function
def collate_fn(batch):
    mix, s1, s2, noise, length = zip(*batch)
    # Pad the sequences to the same length max_len [1,x] -> [1, max_len]
    max_len = max([x.shape[1] for x in mix])
    # max_len = 100000
    # print("max_len", max_len)
    
    # print([x.shape for x in mix])
    mix = [torch.nn.functional.pad(x, (0, max_len - x.shape[1])) for x in mix]
    s1 = [torch.nn.functional.pad(x, (0, max_len - x.shape[1])) for x in s1]
    s2 = [torch.nn.functional.pad(x, (0, max_len - x.shape[1])) for x in s2]
    noise = [torch.nn.functional.pad(x, (0, max_len - x.shape[1])) for x in noise]
    # print([x.shape for x in mix])
    mix = torch.stack(mix)
    s1 = torch.stack(s1)
    s2 = torch.stack(s2)
    noise = torch.stack(noise)
    length = torch.tensor(length)
    return mix.squeeze(), s1.squeeze(), s2.squeeze(), noise.squeeze(), length
    



# Load the dataset
dataset = Libri2Mix("/DATA1/bikash_dutta/CS/SP/A2/LibriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_both.csv")

# Split the dataset
train_set, test_set = torch.utils.data.random_split(dataset, [0.7,0.3])

# Dataloaders
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Testing loop
model.eval()
with torch.inference_mode():
    avg_sdr1 = 0
    avg_sdr2 = 0
    avg_sisnri1 = 0
    avg_sisnri2 = 0
    
    for i, (mix, s1, s2, noise, length) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        mix, s1, s2, noise = mix.to(device), s1.to(device), s2.to(device), noise.to(device)
        # print(f"[LOG] mix shape: {mix.shape}, s1 shape: {s1.shape}, s2 shape: {s2.shape}, noise shape: {noise.shape}")
        est_sources = model.separate_batch(mix)
        sdr1 = sdr_metric(est_sources[:,:,0], s1)
        sdr2 = sdr_metric(est_sources[:,:,1], s2)
        sisnri1 = sisnri_metric(est_sources[:,:,0], s1)
        sisnri2 = sisnri_metric(est_sources[:,:,1], s2)
        # print(f"SDR1: {sdr1}, SDR2: {sdr2}, SISNRi1: {sisnri1}, SISNRi2: {sisnri2}")
        
        # clear memory
        torch.cuda.empty_cache()
        # if i == 10:
        #     break
        avg_sdr1 += sdr1
        avg_sdr2 += sdr2
        avg_sisnri1 += sisnri1
        avg_sisnri2 += sisnri2
    
    print(f"Average SDR1: {avg_sdr1/len(test_dataloader)}, Average SDR2: {avg_sdr2/len(test_dataloader)}, Average SISNRi1: {avg_sisnri1/len(test_dataloader)}, Average SISNRi2: {avg_sisnri2/len(test_dataloader)}")
    wandb.log({
        "pre finetune AVG. SDR1": avg_sdr1/len(test_dataloader),
        "pre finetune AVG. SDR2": avg_sdr2/len(test_dataloader),
        "pre finetune AVG. SISNRi1": avg_sisnri1/len(test_dataloader),
        "pre finetune AVG. SISNRi2": avg_sisnri2/len(test_dataloader)
    })
## fine tune the model

# defineing model class
class SepformerFineTune(torch.nn.Module):
    def __init__(self, model):
        super(SepformerFineTune, self).__init__()
        self.model = model
        # disabling gradient computation
        for parms in self.model.parameters():
            parms.requires_grad = False
        
        # enable gradient computation for the last layer
        named_layers = dict(model.named_modules())
        for name, layer in named_layers.items():
            # print(f"Name: {name}, Layer: {layer}")
            if name == "mods.masknet.output.0":
                for param in layer.parameters():
                    param.requires_grad = True
            if name == "mods.masknet.output_gate":
                for param in layer.parameters():
                    param.requires_grad = True
            

        # printing all tranble parameters
        # for model_name, model_params in model.named_parameters():
        #     print(f"Model Layer Name: {model_name}, Model Params: {model_params.requires_grad}")
    def forward(self, mix):
        est_sources = self.model.separate_batch(mix)
        return est_sources[:,:,0], est_sources[:,:,1] # NOTE: Working with 2 sources ONLY

# Initialize the model
LR = 1e-3
fine_tuned_model = SepformerFineTune(model)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=LR)

EPOCHS = 25
for epoch in tqdm(range(EPOCHS), desc="Epoch Number:"):
    fine_tuned_model.train()
    avg_sdr1 = 0
    avg_sdr2 = 0
    avg_sisnri1 = 0
    avg_sisnri2 = 0
    avg_loss = 0
    for i, (mix, s1, s2, noise, length) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Batch"):
        mix, s1, s2, noise = mix.to(device), s1.to(device), s2.to(device), noise.to(device)
        pred_s1,pred_s2 = fine_tuned_model(mix)
        loss = loss_function(pred_s1, s1) + loss_function(pred_s2, s2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss}")
        torch.cuda.empty_cache()
        avg_sdr1 += sdr_metric(pred_s1, s1)
        avg_sdr2 += sdr_metric(pred_s2, s2)
        avg_sisnri1 += sisnri_metric(pred_s1, s1)
        avg_sisnri2 += sisnri_metric(pred_s2, s2)
        avg_loss += loss
        wandb.log({"train batch_idx":i, "Train Batch Loss": loss, "Train Batch SDR1": sdr_metric(pred_s1, s1), "Train Batch SDR2": sdr_metric(pred_s2, s2), "Train Batch SISNRi1": sisnri_metric(pred_s1, s1), "Train Batch SISNRi2": sisnri_metric(pred_s2, s2)})
    
    print(f"Epoch: {epoch}, Average Loss: {avg_loss/len(train_dataloader)}, Average SDR1: {avg_sdr1/len(train_dataloader)}, Average SDR2: {avg_sdr2/len(train_dataloader)}, Average SISNRi1: {avg_sisnri1/len(train_dataloader)}, Average SISNRi2: {avg_sisnri2/len(train_dataloader)}")
    wandb.log({"Epoch": epoch,"AVG. Loss": {avg_loss/len(train_dataloader)}, "AVG. Train SDR1": avg_sdr1/len(train_dataloader), "AVG. Train SDR2": avg_sdr2/len(train_dataloader), "AVG. Train SISNRi1": avg_sisnri1/len(train_dataloader), "AVG. Train SISNRi2": avg_sisnri2/len(train_dataloader)})
    
    fine_tuned_model.eval()
    with torch.inference_mode():
        avg_sdr1 = 0
        avg_sdr2 = 0
        avg_sisnri1 = 0
        avg_sisnri2 = 0
        avg_loss = 0
        for i, (mix, s1, s2, noise, length) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing Batch"):
            mix, s1, s2, noise = mix.to(device), s1.to(device), s2.to(device), noise.to(device)
            est_sources = fine_tuned_model.model.separate_batch(mix)
            sdr1 = sdr_metric(est_sources[:,:,0], s1)
            sdr2 = sdr_metric(est_sources[:,:,1], s2)
            sisnri1 = sisnri_metric(est_sources[:,:,0], s1)
            sisnri2 = sisnri_metric(est_sources[:,:,1], s2)
            avg_sdr1 += sdr1
            avg_sdr2 += sdr2
            avg_sisnri1 += sisnri1
            avg_sisnri2 += sisnri2
            avg_loss += (loss_function(est_sources[:,:,0], s1) + loss_function(est_sources[:,:,1], s2))
            wandb.log({"test batch_idx":i, "Test Batch SDR1": sdr1, "Test Batch SDR2": sdr2, "Test Batch SISNRi1": sisnri1, "Test Batch SISNRi2": sisnri2})
        
            # save audios to wandb
            if i == 0: # saving only first batch first audio
                
                wandb.log({"Epoch": epoch,"mix":wandb.Audio(mix[0].cpu(),sample_rate=16000), "test_s1": wandb.Audio(s1[0].cpu(),sample_rate=16000),"test_s2": wandb.Audio(s2[0].cpu(),sample_rate=16000),"test_est_s1": wandb.Audio(est_sources[:,:,0][0].cpu(),sample_rate=16000),"test_est_s2": wandb.Audio(est_sources[:,:,1][0].cpu(),sample_rate=16000)})
    
    print(f"Epoch: {epoch}, Average SDR1: {avg_sdr1/len(test_dataloader)}, Average SDR2: {avg_sdr2/len(test_dataloader)}, Average SISNRi1: {avg_sisnri1/len(test_dataloader)}, Average SISNRi2: {avg_sisnri2/len(test_dataloader)}")
    wandb.log({"Epoch": epoch, "AVG. Test SDR1": avg_sdr1/len(test_dataloader), "AVG. Test SDR2": avg_sdr2/len(test_dataloader), "AVG. Test SISNRi1": avg_sisnri1/len(test_dataloader), "AVG. Test SISNRi2": avg_sisnri2/len(test_dataloader)})

# saving checkpoint in dictionary
checkpoint = {
    "model": fine_tuned_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch
}

# Save the checkpoint
torch.save(checkpoint, "fine_tuned_sepformer-wsj03mix.ckpt")


## Test Loading the checkpoint
# Load the checkpoint
checkpoint = torch.load("fine_tuned_sepformer-wsj03mix.ckpt")
fine_tuned_model = SepformerFineTune(model)
fine_tuned_model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
epoch = checkpoint["epoch"]

print(f"EPOCH: {epoch+1}")
print("Checkpoint loaded successfully!")
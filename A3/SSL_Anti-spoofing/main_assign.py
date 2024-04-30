from typing import List
import torch
from custom_dataset import CustomDataset, FOR2SsecDataset
from torch.utils.data import DataLoader
from model import Model
from config import Config
from tqdm import tqdm
from sklearn import metrics
import wandb

import warnings
warnings.filterwarnings('ignore')
from eval_metrics_DF import compute_eer
# making config object
config = Config()

# wandb init
wandb.init(
    project=config.wandb_config['project'],
    config=config.__dict__
)

print("Wandb initialized successfully")
print("Training Configs: ", config.__dict__)

wandb.run.name = config.wandb_config['run_name']

# defining the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# loading weights
model = Model(
    args=config,
    device=device
)
if config.model_path:
    model.load_state_dict(torch.load(config.model_path,map_location=device))

model = model.to(device)

print("Model loaded successfully")


dataset = CustomDataset(dir=config.custom_data_dir, config=config)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=True
)

# print("Data loaded successfully")
# testing data loaders
# for data, target in tqdm(dataloader,total=len(dataloader)):
#     print(data.shape, target.shape)


# evaluating the model on custom dataset
model.eval()
running_auc = 0
running_accuracy = 0
running_eer = 0

pbar = tqdm(enumerate(dataloader),total=len(dataloader),desc="Custom Dataset Evaluation Pre-training")
with torch.no_grad():
    for batch_idx,(data, target) in pbar:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # print(f"[INFO] Output: {output.argmax(dim=-1)}")
        # print(f"[INFO] Target: {target}")
        preds = output.argmax(dim=-1).cpu().numpy()
        target = target.cpu().numpy()
        # print(compute_eer(preds, target))
        fpr, tpr, thresholds = metrics.roc_curve(target, preds, pos_label=1)
        # print(f"[INFO] AUC: {metrics.auc(fpr, tpr)}")
        # print(f"[INFO] Accuracy: {metrics.accuracy_score(target, preds)}")
        running_eer += compute_eer(preds, target)[0]
        running_auc += metrics.auc(fpr, tpr)
        running_accuracy += metrics.accuracy_score(target, preds)
        # logging batch metrics
        wandb.log({
            "pre-training metrics on custom dataset":{"batch_idx": batch_idx, "Batch AUC": metrics.auc(fpr, tpr), "Batch Accuracy": metrics.accuracy_score(target, preds), "Batch EER": compute_eer(preds, target)[0]}
            })
        pbar.set_postfix({"batch_idx": batch_idx, "Batch AUC": metrics.auc(fpr, tpr), "Batch Accuracy": metrics.accuracy_score(target, preds), "Batch EER": compute_eer(preds, target)[0]})
        pbar.refresh()
    # logging epoch metrics
    wandb.log({
        "pre-training metrics on custom dataset":{"AVG AUC": running_auc/len(dataloader), "AVG Accuracy": running_accuracy/len(dataloader), "AVG EER": running_eer/len(dataloader)}
        })
        
    

## fine-tuning the model on FOR dataset

# defining the dataset and dataloader
train_dataset = FOR2SsecDataset(dir=config.for2sec_data_dir, config=config, split="training")
val_dataset = FOR2SsecDataset(dir=config.for2sec_data_dir, config=config, split="validation")
test_dataset = FOR2SsecDataset(dir=config.for2sec_data_dir, config=config, split="testing")

train_dataloader = DataLoader(
    dataset=train_dataset + val_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=True
)

# Note: merging the validation and training dataset
# val_dataloader = DataLoader(
#     dataset=val_dataset,
#     batch_size=config.batch_size,
#     num_workers=config.num_workers,
#     shuffle=False
# )

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=False
)

# testing data loaders
# for data, target in tqdm(train_dataloader,total=len(train_dataloader)):
#     print(data.shape, target.shape)
#     break

# defining the optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

loss_fn = torch.nn.CrossEntropyLoss()

# freezing all the params
for param in model.parameters():
    param.requires_grad = False



# NOTE: Fine-tunning some layers
# enabling the layer using names
for name, param in model.named_parameters():
    # if "encoder" in name or "out_layer" in name or "attention" in name or "LL" in name:
    if "ssl_model" not in name:
        param.requires_grad = True
    # print(name, param.requires_grad)
    


def save_table(my_data:List[List], columns=None, table_name="Audio Table"):

    # create a wandb.Table() with corresponding columns
    if columns is None:
        columns = ["epoch", "batch_id", "audio", "prediction", "truth"]
    
    my_table = wandb.Table(data=my_data, columns=columns)
    wandb.log({table_name: my_table})

# traing the model
my_data_table_for_dataset = []
pbar = tqdm(range(config.num_epochs))
for epoch in pbar:
    model.train()
    # print(f"Epoch: {epoch}")
    running_accuracy = 0
    running_auc = 0
    running_eer = 0
    for batch_idx,(data, target) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training Batches"):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fn(output.softmax(dim=-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = output.argmax(dim=-1).cpu().numpy()
        target = target.cpu().numpy()
        fpr, tpr, thresholds = metrics.roc_curve(target, preds, pos_label=1)
        
        running_accuracy += metrics.accuracy_score(target, preds)
        running_auc += metrics.auc(fpr, tpr)
        running_eer += compute_eer(preds, target)[0]
        
        # logging batch metrics
        wandb.log({
            "training metrics on FOR dataset":{"batch_idx": batch_idx, "Batch Loss": loss.item(), "Batch AUC": metrics.auc(fpr, tpr), "Batch Accuracy": metrics.accuracy_score(target, preds), "Batch EER": compute_eer(preds, target)[0]}
            })
        
    # logging epoch metrics
    wandb.log({
        "training metrics on FOR dataset":{"Epoch": epoch+1, "AVG AUC": running_auc/len(train_dataloader), "AVG Accuracy": running_accuracy/len(train_dataloader), "AVG EER": running_eer/len(train_dataloader)}
        })
    
    pbar.set_postfix({"Epoch": epoch+1, "AVG AUC": running_auc/len(train_dataloader), "AVG Accuracy": running_accuracy/len(train_dataloader), "AVG EER": running_eer/len(train_dataloader)})
    pbar.refresh()
    # evaluating the model on test dataset
    
    model.eval()
    running_accuracy = 0
    running_auc = 0
    running_eer = 0
    with torch.no_grad():
        for batch_idx,(data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Testing Batches"): 
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output.softmax(dim=-1), target)

            preds = output.argmax(dim=-1).cpu().numpy()
            target = target.cpu().numpy()
            fpr, tpr, thresholds = metrics.roc_curve(target, preds, pos_label=1)
            
            running_accuracy += metrics.accuracy_score(target, preds)
            running_auc += metrics.auc(fpr, tpr)
            running_eer += compute_eer(preds, target)[0]
        
            # logging batch metrics
            wandb.log({
                "testing metrics on FOR dataset":{"batch_idx": batch_idx, "Batch Loss": loss.item(), "Batch AUC": metrics.auc(fpr, tpr), "Batch Accuracy": metrics.accuracy_score(target, preds), "Batch EER": compute_eer(preds, target)[0]}
                })
            my_data_table_for_dataset += [[epoch ,batch_idx, wandb.Audio(audio,sample_rate=16000), pred, target] for (audio,pred,target) in zip(data.cpu()[:5].numpy(), preds[:5], target[:5])]
            




        # logging epoch metrics
        wandb.log({
            "testing metrics on FOR dataset":{"Epoch": epoch+1, "AVG AUC": running_auc/len(test_dataloader), "AVG Accuracy": running_accuracy/len(test_dataloader), "AVG EER": running_eer/len(test_dataloader)}
            })
    pbar.set_postfix({"Epoch": epoch+1, "AVG AUC": running_auc/len(test_dataloader), "AVG Accuracy": running_accuracy/len(test_dataloader), "AVG EER": running_eer/len(test_dataloader)})
    pbar.refresh()

# testing the model on custom dataset after fine-tuning
model.eval()
running_auc = 0
running_accuracy = 0
running_eer = 0
my_data_table_custom_dataset = []
with torch.no_grad():
    for batch_idx,(data, target) in tqdm(enumerate(dataloader),total=len(dataloader),desc="Custom Dataset Evaluation Post-training"):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # print(f"[INFO] Output: {output.argmax(dim=-1)}")
        # print(f"[INFO] Target: {target}")
        preds = output.argmax(dim=-1).cpu().numpy()
        target = target.cpu().numpy()
        # print(compute_eer(preds, target))
        fpr, tpr, thresholds = metrics.roc_curve(target, preds, pos_label=1)
        # print(f"[INFO] AUC: {metrics.auc(fpr, tpr)}")
        # print(f"[INFO] Accuracy: {metrics.accuracy_score(target, preds)}")
        running_eer += compute_eer(preds, target)[0]
        running_auc += metrics.auc(fpr, tpr)
        running_accuracy += metrics.accuracy_score(target, preds)
        # logging batch metrics
        wandb.log({
            "post-training metrics on custom dataset":{"batch_idx": batch_idx, "Batch AUC": metrics.auc(fpr, tpr), "Batch Accuracy": metrics.accuracy_score(target, preds), "Batch EER": compute_eer(preds, target)[0]}
            })
        # creating a table with audio pred and actual labels
        my_data_table_custom_dataset += [[epoch ,batch_idx, wandb.Audio(audio,sample_rate=16000), pred, target] for (audio,pred,target) in zip(data.cpu()[:25].numpy(), preds[:25], target[:25])]
        
    # logging epoch metrics
    wandb.log({
        "post-training metrics on custom dataset":{"AVG AUC": running_auc/len(dataloader), "AVG Accuracy": running_accuracy/len(dataloader), "AVG EER": running_eer/len(dataloader)}
        })


save_table(my_data_table_for_dataset, table_name="FOR Dataset Table")
save_table(my_data_table_custom_dataset, table_name="Custom Dataset Table")

# saving the model
model_cheackpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "loss_fn": loss_fn.state_dict(),
    "epoch": epoch
}

torch.save(model_cheackpoint,"for_trained_model.ckpt")

wandb.finish()
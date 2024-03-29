import torch
from ctc import ctc_loss
from torch.nn import functional as F

log_probs = torch.randn(50, 16, 20).log_softmax(2)
targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
blank = 0

# Custom CTC Loss
custom_ctc_loss = ctc_loss
custom_loss = custom_ctc_loss(log_probs, targets, input_lengths, target_lengths)

# PyTorch CTC Loss
torch_ctc_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank, zero_infinity=True)

print("Custom CTC Loss:", custom_loss.item())
print("PyTorch CTC Loss:", torch_ctc_loss.item())
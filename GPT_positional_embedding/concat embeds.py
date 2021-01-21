import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 3
max_len = 450
dim = 768

tr_input_length = 68134
train_prob_embeds = torch.zeros((tr_input_length, max_len, dim))

idx = 0
while True:
    file_name = 'train_prob_embeds_'+str(idx)+'_.pt'
    tmp = torch.load(file_name)
    train_prob_embeds[idx:idx + len(tmp), :] = tmp
    idx+=32
#
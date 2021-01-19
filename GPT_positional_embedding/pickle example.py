import pickle
import torch
device = torch.device('cpu')

file = open('../dev_probs.txt', 'rb')
file = list(file)
for i in range(len(file)):
    f = file[i]
    q = torch.load(open(f,'r'), map_location={'cuda:0':'cpu'})

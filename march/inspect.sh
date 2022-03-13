import torch
m=torch.load('model_10000.p')
for k,v in m.items():
    print(k,v.shape)
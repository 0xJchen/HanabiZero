import torch
import torch.nn as nn
class PreActResBlock(nn.Module):
    def __init__(self, in_dim):
        super(PreActResBlock,self).__init__()
        self.in_dim = in_dim

        self.bn1=nn.BatchNorm1d(in_dim)
        self.fc1=nn.Linear(in_dim, in_dim)

        self.bn2=nn.BatchNorm1d(in_dim)
        self.fc2=nn.Linear(in_dim, in_dim)

    def forward(self, x):
        _x = x

        x=self.bn1(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)

        x=self.bn2(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        x+=_x

        return x
class PreActResTower(nn.Module):
    
    def __init__(self, in_dim, layer):
        super(PreActResTower,self).__init__()
        layers=[PreActResBlock(in_dim) for _ in range(layer)]
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        x=self.layers(x)
        return x
def cnt(md):
    return sum(p.numel() for p in md.parameters())

block1=PreActResBlock(512)
print("dim=512, resblock ={}".format(cnt(block1)))
tower1=PreActResTower(512,2)
print("dim=512, tower of 2 blocks = {}".format(cnt(tower1)))
import torch
import torch.nn as nn
torch.set_printoptions(profile='full')
model=torch.load("recent/model/model_90000.p")
#extract dynamics_reward
def ifnan(t):
    return torch.isnan(t).any()
need_k=[]
dict_model=dict(model)
mock_next_state=torch.load('/home/game/revert/lsy/lst_best_confirm_copy/HanabiZero/72/next_state_41')
for k in dict_model.keys():
    if 'dynamics_reward' in k:
        p=torch.tensor(dict_model[k])
        print(k,p.shape,torch.isnan(p).any(),p.max(),p.min())
        need_k.append(k)
        # if '0' in k:
            # print(p)
print(need_k)
weight=torch.tensor(dict_model[need_k[0]])
bias=torch.tensor(dict_model[need_k[1]])
a0=torch.matmul(weight,mock_next_state[41])
print("[shape] weight={},input={}".format(weight.shape,mock_next_state[41].shape))
print("[nan]weight={},input={},output={}".format(ifnan(weight),ifnan(mock_next_state[41]),ifnan(a0)))
print("[stat]weight max={},min={}; input max={}, min={}".format(weight.max(),weight.min(),mock_next_state[41].float().max(),mock_next_state[41].min()))
print(65504<mock_next_state[41].max())
for id,i in enumerate(mock_next_state[41]):
    if i>=65500:
        # print('gg')
        print(id,i)
# print("weight=",weight)
# print("input=",mock_next_state[41])
# print(torch.isnan(a0).any())
# a1=a0+bias
# print(torch.isnan(a1).any())
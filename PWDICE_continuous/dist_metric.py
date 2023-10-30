import torch
from NN import *
import torch.nn.functional as F 
device = torch.device('cuda:0')

def cosine_similarity(X, Y):
    return 

def get_dist(X, Y, scale=1, args_distance='no', dist_model=None): # dist_model is a tuple, the first element is model, the second elemnet is args
            # X, Y are tensors of states (batch size * state dim) 
            if args_distance == "squared_euclidean": V = ((X - Y) ** 2).sum(dim=1).view(-1, 1) # now only Euclidean
            elif args_distance == "euclidean": V = (((X - Y) ** 2).sqrt().sum(dim=1)).view(-1, 1)
            elif args_distance == "manhattan": V = (X - Y).abs().sum(dim=1).view(-1, 1)
            elif args_distance == "exped-manhattan": V = ((( X - Y).abs().exp().sum(dim=1)) - 1).view(-1, 1)
            elif args_distance == "dirac": V = ((X - Y).abs() < 1e-5).double()
            elif args_distance == "cosine": V = 1 - torch.nn.CosineSimilarity(dim=1, eps=1e-6)(X, Y).view(-1, 1)
            elif args_distance == "special_hopper": 
                idx1, idx2 = [0, 5, 6], [1, 2, 3, 4, 7, 8, 9, 10]
                V1 = ((X[:, idx1] - Y[:, idx1]) ** 2).sum(dim=1).view(-1, 1)
                V2 = (1 - torch.nn.CosineSimilarity(dim=1, eps=1e-6)(X[:, idx2], Y[:, idx2])).view(-1, 1)
                # print(V1.shape, V2.shape)
                #print("V1:", V1[:100].view(-1),"V2:", V2[:100].view(-1))
                V = (V1 + V2)
            elif args_distance == "special_pendulum": 
                V = ((1 - torch.nn.CosineSimilarity(dim=1, eps=1e-6)(X[:, :-1], Y[:, :-1])) + ((X[:, -1] - Y[:, -1]) / 8) ** 2).view(-1, 1)
                # print("shape:", torch.nn.CosineSimilarity(dim=1, eps=1e-6)(X[:, :-1], Y[:, :-1]).shape, (((X[:, -1] - Y[:, -1]) / 8) ** 2).shape)
            elif args_distance == "special_lunarlander":
                idx = [0, 1, 2, 3, 5, 6, 7]
                V1 = (X[:, idx] - Y[:, idx]).abs().sum(dim=1).view(-1, 1)
                V2 = (1 - torch.nn.CosineSimilarity(dim=1, eps=1e-6)(torch.cat([torch.sin(X[:, 4]).view(-1, 1), torch.cos(X[:, 4]).view(-1, 1)], dim=1), torch.cat([torch.sin(Y[:, 4]).view(-1, 1), torch.cos(Y[:, 4]).view(-1, 1)], dim=1))).view(-1, 1)
                # print(V1.shape, V2.shape)
                #print("V1:", V1[:100].view(-1),"V2:", V2[:100].view(-1))
                V = (V1 + V2)
            elif args_distance.find("learned") != -1: 
                assert dist_model is not None, "dist_model is None!"
                #if isinstance(dist_model, Contrastive) or isinstance(dist_model, Contrastive_PD): V = ((dist_model.encode(XX) - dist_model.encode(YY)) ** 2).sum(dim=1).view(-1, 1)
                #elif isinstance(dist_model, Contrastive_Sphere): 
                # V = (1 - F.cosine_similarity(XX, YY)).view(-1, 1)
                if args_distance.find("cos") != -1:
                    V = (1 - (dist_model.encode(X) * dist_model.encode(Y)).sum(dim=1)).view(-1, 1).detach()
                else: V = ((dist_model.encode(X) - dist_model.encode(Y)) ** 2).sum(dim=1).view(-1, 1).detach() 
            else: raise NotImplementedError("Distance type error!")
            
            return V * scale 
            
def soft_linear_piecewise_loss(x, target, lenient):
    if x < target - lenient: return target - x
    elif x < target + lenient: return (target + lenient - x) ** 2 / (4 * lenient)
    else: return torch.zeros(1).double().to(device)
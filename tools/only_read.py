import pickle
import torch
import random
import torch.nn.functional as F
# specify the file path
# file_path = "/mnt/data/deka01/debug_OpenPCDet/tools/sim_scores.pkl"
# device = torch.device('cuda:1')
# with open(file_path, 'rb') as f:
#     serialized_object = torch.load(f, map_location=device)
#     print(serialized_object)
# file_path = '/mnt/data/deka01/debug_OpenPCDet/tools/cos_scores.pkl'
# open the file for reading in binary mode
file_path = '/mnt/data/deka01/debug_OpenPCDet/tools/ema_16900.9.pkl'
with open(file_path, "rb") as file:
    # load the contents of the file using pickle
    data = pickle.load(file)
car_ped = F.cosine_similarity(data['Car']['mean']['sh'].unsqueeze(0),data['Ped']['mean']['sh'].unsqueeze(0))
print("car vs ped",car_ped)
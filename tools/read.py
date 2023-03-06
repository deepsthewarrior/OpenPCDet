import pickle
import torch
import random
# specify the file path
# file_path = "/mnt/data/deka01/debug_OpenPCDet/tools/shared_corrected.pkl"

file_path = '/mnt/data/deka01/debug_OpenPCDet/tools/feature_bank.pkl'
# open the file for reading in binary mode
with open(file_path, "rb") as file:
    # load the contents of the file using pickle
    data = pickle.load(file)

# print the data
# print(data)
# pkl_keys = ['Car_cls_ema','Car_reg_ema','Car_sh_ema','Ped_cls_ema','Ped_reg_ema','Ped_sh_ema','Cyc_cls_ema','Cyc_reg_ema','Cyc_sh_ema']
feature = {}

car_cls = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_cls'][0].device)
car_sh = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_sh'][0].device)
car_reg = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_reg'][0].device)
car_cls_mean = car_cls.squeeze(dim=0) 
car_sh_mean = car_sh.squeeze(dim=0)
car_reg_mean = car_reg.squeeze(dim=0)

ped_cls = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_cls'][0].device)
ped_sh = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_sh'][0].device)
ped_reg = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_reg'][0].device)
ped_cls_mean = ped_cls.squeeze(dim=0)
ped_sh_mean = ped_sh.squeeze(dim=0)
ped_reg_mean = ped_reg.squeeze(dim=0)

cyc_cls = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_cls'][0].device)
cyc_sh = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_sh'][0].device)
cyc_reg = torch.zeros([data['Car_cls'][0].shape[0]],device=data['Car_reg'][0].device)
cyc_cls_mean = cyc_cls.squeeze(dim=0) 
cyc_sh_mean = cyc_sh.squeeze(dim=0)
cyc_reg_mean = cyc_reg.squeeze(dim=0)

Car_cls = data['Car_cls']
Car_reg = data['Car_reg']
Car_sh = data['Car_sh']
Ped_cls = data['Ped_cls']
Ped_reg = data['Ped_reg']
Ped_sh = data['Ped_sh']
Cyc_cls = data['Cyc_cls']
Cyc_reg = data['Cyc_reg']
Cyc_sh = data['Cyc_sh']

alpha = 0.9

for i,cls_feat in enumerate(Car_cls):
    car_cls = cls_feat*(1-alpha) + car_cls*(alpha)
    car_sh = Car_sh[i]*(1-alpha) + car_sh*(alpha)
    car_reg = Car_sh[i]*(1-alpha) + car_sh*(alpha)
    
for  i,cls_feat in enumerate(Ped_cls):
    ped_cls = cls_feat*(1-alpha) + ped_cls*(alpha)
    ped_sh = Ped_sh[i]*(1-alpha) + ped_sh*(alpha)
    ped_reg = Ped_sh[i]*(1-alpha) + ped_sh*(alpha)

for  i,cls_feat in enumerate(Cyc_cls):
    cyc = cls_feat*(1-alpha) + cyc_cls*(alpha)
    cyc_sh = Cyc_sh[i]*(1-alpha) + cyc_sh*(alpha)
    cyc_reg = Cyc_sh[i]*(1-alpha) + cyc_sh*(alpha)


Car_cls_sq =  [t.squeeze(dim=0) for t in Car_cls]
Car_reg_sq = [t.squeeze(dim=0) for t in Car_reg]
Car_sh_sq = [t.squeeze(dim=0) for t in Car_sh]
Car_cls_mean = torch.mean(torch.stack(Car_cls_sq),dim=0)
Car_reg_mean = torch.mean(torch.stack(Car_reg_sq),dim=0)
Car_sh_mean = torch.mean(torch.stack(Car_sh_sq),dim=0)

Ped_cls_sq =  [t.squeeze(dim=0) for t in Ped_cls]
Ped_reg_sq = [t.squeeze(dim=0) for t in Ped_reg]
Ped_sh_sq = [t.squeeze(dim=0) for t in Ped_sh]
Ped_cls_mean = torch.mean(torch.stack(Ped_cls_sq),dim=0)
Ped_reg_mean = torch.mean(torch.stack(Ped_reg_sq),dim=0)
Ped_sh_mean = torch.mean(torch.stack(Ped_sh_sq),dim=0)

Cyc_cls_sq =  [t.squeeze(dim=0) for t in Cyc_cls]
Cyc_reg_sq = [t.squeeze(dim=0) for t in Cyc_reg]
Cyc_sh_sq = [t.squeeze(dim=0) for t in Cyc_sh]
Cyc_cls_mean = torch.mean(torch.stack(Cyc_cls_sq),dim=0)
Cyc_reg_mean = torch.mean(torch.stack(Cyc_reg_sq),dim=0)
Cyc_sh_mean = torch.mean(torch.stack(Cyc_sh_sq),dim=0)

feature['Car'] = {}
feature['Ped'] = {}
feature['Cyc'] = {}

feature['Car']['ema'] = {
    'cls':car_cls,
    'reg':car_reg,
    'sh':car_sh,
    }
feature['Car']['mean'] = {
    'cls':Car_cls_mean,
    'reg':Car_reg_mean,
    'sh':Car_sh_mean,
}

feature['Ped']['ema'] = {
    'cls':ped_cls,
    'reg':ped_reg,
    'sh':ped_sh,
    }
feature['Ped']['mean'] = {
    'cls':Ped_cls_mean,
    'reg':Ped_reg_mean,
    'sh':Ped_sh_mean,
}

feature['Cyc']['ema'] = {
    'cls':cyc_cls,
    'reg':cyc_reg,
    'sh':cyc_sh,
    }
feature['Cyc']['mean'] = {
    'cls':Cyc_cls_mean,
    'reg':Cyc_reg_mean,
    'sh':Cyc_sh_mean,
}



# pkl_dict={'ens':[]}
# for i,val in enumerate(data['ens']):
#     pkl_dict['ens'].append(val)
#     if i==4:
#      break
fl_name = "ema_" + str(alpha) + ".pkl"
with open(fl_name, "wb") as f:
    pickle.dump(feature,f)
        
# feature_keys = ['Car_cls','Ped_cls','Cyc_cls','Car_reg','Car_sh','Ped_reg','Ped_sh','Cyc_reg','Cyc_sh']
# features = {x:[] for x in feature_keys}
# ens_list = data['ens']
# count = [0,0,0]
# car = torch.zeros([1, ens_list[0]['rcnn_cls_interim'].shape[1]])
# cyc = None


# ## for bank
# for val in data['ens']:
#     pred_scores = val['pred_scores']
#     pred_labels = val['pred_labels']
#     selected = val['selected']
#     iou = val['iou']
#     gt_labels = val['gt_label']
#     gt_assign = val['gt_assignment']
#     selected_gt = gt_assign[selected]
#     selected_gt_labels = gt_labels[selected_gt]
#     tp_cls = pred_labels[selected] == selected_gt_labels 
#     iou_selected = iou[selected] > 0.80

#     final_sel = iou_selected & tp_cls
#     final_preds = pred_labels[selected][final_sel]
#     final_shared = val['shared_features'][selected][final_sel]
#     cls_inter = val['rcnn_cls_interim'][selected][final_sel]
#     reg_inter = val['rcnn_reg_interim'][selected][final_sel]
    
#     car_cls = cls_inter[final_preds == 1]
#     car_reg = reg_inter[final_preds == 1]
#     car_sh = final_shared[final_preds == 1]
#     for i,feature in enumerate(car_cls):
#         features['Car_cls'].append(feature)
#         features['Car_reg'].append(car_reg[i])
#         features['Car_sh'].append(car_sh[i])
        
#     ped_cls = cls_inter[final_preds == 2]
#     ped_reg = reg_inter[final_preds == 2]
#     ped_sh = final_shared[final_preds == 2]
#     for i,feature in enumerate(ped_cls):
#         features['Ped_cls'].append(feature)
#         features['Ped_reg'].append(ped_reg[i])
#         features['Ped_sh'].append(ped_sh[i])

#     cyc_cls = cls_inter[final_preds == 3]
#     cyc_reg = reg_inter[final_preds == 3]
#     cyc_sh = final_shared[final_preds == 3]
#     for i,feature in enumerate(cyc_cls):
#         features['Cyc_cls'].append(feature)
#         features['Cyc_reg'].append(cyc_reg[i])
#         features['Cyc_sh'].append(cyc_sh[i])    
    
#     with open('feature_bank_part.pkl','wb') as f:
#         pickle.dump(features,f)
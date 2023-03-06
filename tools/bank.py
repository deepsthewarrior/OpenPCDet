import pickle
import torch

file_path = "/mnt/data/deka01/debug_OpenPCDet/tools/feature_bank_part.pkl"
# open the file for reading in binary mode
with open(file_path, "rb") as file:
    # load the contents of the file using pickle
    data = pickle.load(file)

for x in data:
    continue

# # print the data
# # print(data)
# feature_keys = ['Car_cls','Ped_cls','Cyc_cls','Car_reg','Car_sh','Ped_reg','Ped_sh','Cyc_reg','Cyc_sh']
# features = {x:[] for x in feature_keys}
# ens_list = data['ens']
# # count = [0,0,0]
# # car = torch.zeros([1, ens_list[0]['rcnn_cls_interim'].shape[1]])
# # cyc = None

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
    
#     with open('feature_bank.pkl','wb') as f:
#         pickle.dump(features,f)
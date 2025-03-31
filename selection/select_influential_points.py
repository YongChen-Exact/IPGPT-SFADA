import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
import torch.nn as nn
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from selection.stratrgy import select_IPL
from dataloaders.dataset import BaseDataSets, DataSets_like, RandomGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def Savefeat():
    class_num = 2
    db_source_like = DataSets_like(base_dir="/home/data/CY/Datasets/MMS/Heart_B", split="train",
                                   transform=RandomGenerator([256, 256]))
    source_like_loader = DataLoader(db_source_like, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    db_target = BaseDataSets(base_dir="/home/data/CY/Datasets/MMS/Heart_B", split="train",
                             transform=RandomGenerator([256, 256]))
    target_loader = DataLoader(db_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    model = net_factory(net_type="unet", in_chns=1, class_num=class_num)
    model.load_state_dict(
        torch.load('/home/data/CY/codes/IPGPT-SFADA/Model/Heart_A_unet/unet_best_model.pth')["state_dict"])
    class_features = Class_Features(numbers=class_num)
    print("source-like sample num:", len(source_like_loader))
    print("target sample num:", len(target_loader))
    s_reprs = []
    s_logits = []
    s_inds = []
    s_labels = []

    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(source_like_loader):
            image_batch, label_batch, name = (sampled_batch["image"], sampled_batch["label"], sampled_batch["name"])
            image_batch, labels = (image_batch.cuda(), label_batch.cuda())
            model.eval()
            feat_cls, output = model(image_batch)
            vectors = class_features.calculate_mean_vector(feat_cls, output)
            feat_vectors = np.zeros([class_num, 256])
            for t in range(class_num):
                feat_vectors[t] = vectors[t].detach().cpu().numpy().squeeze()
            single_vectors = feat_vectors.astype(float)
            s_reprs.append(single_vectors)
            s_logits.append(output.cpu())
            s_inds.append(batch_idx)
            label = label_batch.reshape(-1).detach().cpu().numpy()
            s_labels.append(label)

    t_reprs = []
    t_logits = []
    t_inds = []
    t_idx_names = {}
    t_lenth = len(target_loader)
    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(target_loader):
            image_batch, label_batch, name = (sampled_batch["image"], sampled_batch["label"], sampled_batch["name"])
            image_batch, labels = (image_batch.cuda(), label_batch.cuda())
            t_idx_names[batch_idx] = name
            model.eval()
            feat_cls, output = model(image_batch)
            vectors = class_features.calculate_mean_vector(feat_cls, output)
            single_image_objective_vectors = np.zeros([class_num, 256])
            for t in range(class_num):
                single_image_objective_vectors[t] = vectors[t].detach().cpu().numpy().squeeze()
            single_vectors = single_image_objective_vectors.astype(float)
            t_reprs.append(single_vectors)
            t_logits.append(output.cpu())
            t_inds.append(batch_idx)
    k = 10
    selected_rate = 0.1
    selected_num = int(t_lenth * selected_rate)
    print("selected_num:", selected_num)
    selected_list = select_IPL(k, selected_num, s_reprs, s_logits, s_labels, t_reprs, t_logits, t_inds)
    selected_cases = []
    for idx in selected_list:
        selected_cases.append(str(t_idx_names[idx])[2:-2])
    file = os.path.join('/home/data/CY/codes/IPGPT-SFADA/Code_OA/Results/selection_list', f'Heart_B.txt')
    with open(file, 'w') as file:
        for i in range(len(selected_cases)):
            img = str(selected_cases[i])
            img = img.strip("[]'")
            file.write(img + '\n')
    print(f"File saved: {file}")


class Class_Features:
    def __init__(self, numbers=19):
        self.class_numbers = numbers
        return

    def calculate_mean_vector(self, feat_cls, outputs):
        outputs_softmax = F.softmax(outputs, dim=1)
        scale_factor = F.adaptive_avg_pool2d(outputs_softmax, 1)
        vectors = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                s = feat_cls[n] * outputs_softmax[n][t]
                s = torch.mean(s, dim=0).unsqueeze(0)
                max_pool = nn.MaxPool2d(kernel_size=16)
                output = max_pool(s)
                s = output.view(output.size(0), -1) / scale_factor[n][t]
                vectors.append(s)
        return vectors


if __name__ == "__main__":
    Savefeat()

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
import torch.nn as nn
import math
from networks.net_factory import net_factory
from torch.utils.data import DataLoader
from selection.stratrgy import select_IPL
from dataloaders.dataset import BaseDataSets, DataSets_like, RandomGenerator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def select(FLAGS):
    db_source_like = DataSets_like(base_dir=FLAGS.target_path, split="train", transform=RandomGenerator([256, 256]))
    source_like_loader = DataLoader(db_source_like, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    db_target = BaseDataSets(base_dir=FLAGS.target_path, split="train", transform=RandomGenerator([256, 256]))
    target_loader = DataLoader(db_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    model = net_factory(net_type="unet", in_chns=1, class_num=FLAGS.num_classes)
    model.load_state_dict(torch.load(FLAGS.checkpoint)["state_dict"])
    class_features = Class_Features(numbers=FLAGS.num_classes)
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
            feat_vectors = np.zeros([FLAGS.num_classes, 256])
            for t in range(FLAGS.num_classes):
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
    uncertainties = []
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
            single_image_objective_vectors = np.zeros([FLAGS.num_classes, 256])
            for t in range(FLAGS.num_classes):
                single_image_objective_vectors[t] = vectors[t].detach().cpu().numpy().squeeze()
            single_vectors = single_image_objective_vectors.astype(float)
            t_reprs.append(single_vectors)
            t_logits.append(output.cpu())
            t_inds.append(batch_idx)

            mc_iteration = 10
            all_predictions = []
            model.train()
            for _ in range(mc_iteration):
                _, out = model(image_batch)
                probs = torch.softmax(out, dim=1)
                all_predictions.append(probs.unsqueeze(0))
            all_predictions = torch.cat(all_predictions, dim=0)
            uncertainty = torch.var(all_predictions, dim=0).squeeze(0)
            uncertainties.append(uncertainty.cpu())
    uncertainties = np.array(uncertainties)
    print(uncertainties.shape)
    selected_num = int(t_lenth * FLAGS.selected_rate)
    print("selected_num:", selected_num)
    selected_list = select_IPL(selected_num, FLAGS.num_classes, s_reprs, s_logits, t_reprs, t_logits, t_inds, uncertainties, int(math.sqrt(len(target_loader))), FLAGS.tau)
    selected_cases = []
    for idx in selected_list:
        selected_cases.append(str(t_idx_names[idx])[2:-2])
    list_name = FLAGS.target_path.split('/')[-1]+'.txt'
    file = os.path.join(FLAGS.save_list_path, list_name)
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
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


parser = argparse.ArgumentParser()
parser.add_argument("--target_path", type=str, default="/home/data/CY/Datasets/Prostate_HCRUDB",
                    help="Path to the target dataset")
parser.add_argument('--checkpoint', type=str,
                    default='/home/data/CY/code/IPGPT-SFADA/Model/Prostate_RUNMC_BMC_unet/unet_best_model.pth',
                    help='Path to the pre-trained source model checkpoint')
parser.add_argument('--selected_rate', type=float, default=0.1,
                    help='Percentage of influential points to be selected')
parser.add_argument('--tau', type=float, default=0.99,
                    help='Similarity threshold for diversity-aware filtering')
parser.add_argument('--save_list_path', type=str, default='/home/data/CY/code/IPGPT-SFADA/Results/selection_list',
                    help='Directory to save the list of selected source-like samples')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of output classes (network output channels)')


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    select(FLAGS)

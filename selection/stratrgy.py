import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
from tqdm import tqdm
import torch.nn as nn


def compute_dynamic_K(targets, source_pool, k0=10, K_max=30):
    nbrs = NearestNeighbors(n_neighbors=k0).fit(source_pool)
    distances, _ = nbrs.kneighbors(targets)
    avg_dist = distances[:, 1:].mean(axis=1)
    density = 1 / (avg_dist + 1e-6)
    rho_norm = (density - density.min()) / (density.max() - density.min())
    k_min_auto = np.clip(int(K_max * (1 - rho_norm.mean())), 1, K_max // 2)
    k_dynamic = (K_max - rho_norm * (K_max - k_min_auto)).astype(int)
    return k_dynamic


def select_IPL(size, class_num, s_reprs, s_logits, t_reprs, t_logits, t_inds, uncertainties, K_max, tau):
    s_reprs, t_reprs = torch.tensor(s_reprs), torch.tensor(t_reprs)
    s_reprs = s_reprs.view(s_reprs.size(0), -1)
    t_reprs = t_reprs.view(t_reprs.size(0), -1)
    nbrs = NearestNeighbors(n_neighbors=K_max).fit(s_reprs.numpy())
    criterion = nn.KLDivLoss(reduction='none')
    kl_scores = []
    selected_inds = []
    K_dynamic = compute_dynamic_K(t_reprs.numpy(), s_reprs.numpy(), K_max=K_max)
    for unlab_i, (target_repr, target_logit) in enumerate(tqdm(zip(t_reprs, t_logits), desc="Finding neighbours for every unlabeled data point")):
        k_i = int(K_dynamic[unlab_i])
        distances_, neighbors = nbrs.kneighbors(X=target_repr.view(1, -1).numpy(), n_neighbors=k_i)
        neigh_prob = [F.softmax(s_logits[n], dim=1) for n in neighbors[0]]
        candidate_prob = torch.softmax(target_logit, dim=1)
        candidate_log_prob = F.log_softmax(target_logit, dim=1)

        pseudo_label = candidate_prob.argmax(dim=1)
        conf_mask = candidate_prob.max(dim=1).values > 0.8
        r_c = torch.zeros(class_num)
        valid_class_mask = torch.zeros(class_num, dtype=torch.bool)
        for c in range(class_num):
            class_mask = (pseudo_label == c) & conf_mask
            class_area = class_mask.float().sum()
            total_area = conf_mask.float().sum()
            if class_area > 0:
                r_c[c] = class_area / (total_area + 1e-8)
                valid_class_mask[c] = True

        for c in range(class_num):
            mask = (pseudo_label == c)
            if mask.sum() > 0:
                unc = uncertainties[unlab_i][c,:,:]
                class_uncert = unc.mean()
                r_c[c] *= torch.exp(class_uncert)

        weight = torch.ones(class_num)
        weight[valid_class_mask] = 1 / (r_c[valid_class_mask] + 1e-8)
        weighted_kl_per_neigh = []
        for n_prob in neigh_prob:
            kl = criterion(candidate_log_prob, n_prob)
            kl = torch.sum(kl, dim=1, keepdim=True)
            weight_map = weight[pseudo_label]
            weighted_kl = kl * weight_map.unsqueeze(1)
            weighted_kl_per_neigh.append(weighted_kl)

        mean_weighted_kl = torch.mean(torch.stack(weighted_kl_per_neigh), dim=0)
        kl_scores.append(mean_weighted_kl.mean())

        is_redundant = False
        for selected_idx in selected_inds:
            candidate_vector = target_repr
            selected_vector = t_reprs[selected_idx]
            cosine_sim = np.dot(candidate_vector, selected_vector.T) / (
                    np.linalg.norm(candidate_vector) * np.linalg.norm(selected_vector))
            if cosine_sim > tau:
                is_redundant = True
                break
        if not is_redundant:
            selected_inds.append(unlab_i)
    selected_kl_scores = [kl_scores[i] for i in selected_inds]
    selected_inds_with_kl = np.argsort(selected_kl_scores)[-size:]
    return list(np.array(t_inds)[selected_inds_with_kl])


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

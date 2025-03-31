import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)
from tqdm import tqdm
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier


def select_IPL(k, size, s_reprs, s_logits, s_labels, t_reprs, t_logits, t_inds):
    neigh = KNeighborsClassifier(n_neighbors=k)
    s_reprs, t_reprs = torch.tensor(s_reprs), torch.tensor(t_reprs)
    s_reprs = s_reprs.view(s_reprs.size(0), -1)
    t_reprs = t_reprs.view(t_reprs.size(0), -1).unsqueeze(1)
    neigh.fit(X=s_reprs, y=np.array(s_labels))
    criterion = nn.KLDivLoss(reduction='none')
    kl_scores = []
    distances = []
    selected_inds = []
    for unlab_i, candidate in enumerate(
            tqdm(zip(t_reprs, t_logits), desc="Finding neighbours for every unlabeled data point")):
        distances_, neighbours = neigh.kneighbors(X=candidate[0], return_distance=True)
        distances.append(distances_[0])
        neigh_prob = [F.softmax(s_logits[n], dim=1) for n in neighbours[0]]
        candidate_log_prob = F.log_softmax(candidate[1], dim=-1)
        kl = np.array([torch.sum(criterion(candidate_log_prob, n), dim=-1).numpy() for n in neigh_prob])
        kl_scores.append(kl.mean())
        threshold = 0.990
        is_redundant = False
        for selected_idx in selected_inds:
            candidate_vector = candidate[0]
            selected_vector = t_reprs[selected_idx]
            cosine_sim = np.dot(candidate_vector, selected_vector.T) / (
                    np.linalg.norm(candidate_vector) * np.linalg.norm(selected_vector))
            if cosine_sim > threshold:
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

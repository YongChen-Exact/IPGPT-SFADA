import argparse
import os
import h5py
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def single_volume_fast(case, net, FLAGS, patch_size=[256, 256]):
    h5f = h5py.File(FLAGS.target_path + "/{}".format(case), 'r')
    image = h5f['image'][:]
    x, y = image.shape[0], image.shape[1]
    zoomed_slices = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(zoomed_slices).unsqueeze(0).unsqueeze(0).float().cuda()
    net.train()
    mc_iteration = 10
    all_predictions = []
    for _ in range(mc_iteration):
        with torch.no_grad():
            _, out = net(input)
            probs = torch.softmax(out, dim=1)
            all_predictions.append(probs.unsqueeze(0))
    all_predictions = torch.cat(all_predictions, dim=0)
    if FLAGS.uncertainty == 'variance':
        variance = torch.var(all_predictions, dim=0).squeeze(0)
        uncertainty = variance[1].mean().item()
    else:
        mean_probs = all_predictions.mean(dim=0).squeeze(0)
        image_entropy = -torch.sum(mean_probs[1] * torch.log(mean_probs[1] + 1e-8), dim=0)
        entropy = torch.mean(image_entropy)
        uncertainty = entropy.cpu().numpy()
    return uncertainty


def Inference(FLAGS):
    image_list = sorted(os.listdir(FLAGS.target_path))
    snapshot_path = FLAGS.source_model
    destination_path = FLAGS.source_like_samples+FLAGS.uncertainty
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    net = net_factory(net_type='unet', in_chns=1,
                      class_num=FLAGS.num_classes)
    net.load_state_dict(torch.load(snapshot_path)["state_dict"])
    print("init weight from {}".format(snapshot_path))
    uncertainties = []
    for case in tqdm(image_list):
        uncertainty = single_volume_fast(case, net, FLAGS)
        uncertainties.append(uncertainty)

    percentile = 50  # percentage of source-like sample
    threshold = np.percentile(uncertainties, percentile)
    filtered_indices = [i for i, uncertainty in enumerate(uncertainties) if uncertainty < threshold]
    filtered_image_names = [image_list[i] for i in filtered_indices]
    for file in filtered_image_names:
        source_file = os.path.join(FLAGS.target_path, file)
        destination_file = os.path.join(destination_path, file)
        shutil.copy2(source_file, destination_file)


parser = argparse.ArgumentParser()
parser.add_argument("--target_path", type=str, default="/home/data/CY/Datasets/Prostate_HCRUDB/training_set", help="Name of Experiment")
parser.add_argument('--source_model', type=str,
                    default='/home/data/CY/codes/IPGPT-SFADA/Model/Prostate_HCRUDB_unet/unet_best_model.pth',
                    help='data_name')
parser.add_argument('--uncertainty', type=str, default='variance', help='entropy or variance')
parser.add_argument('--source_like_samples', type=str, default='/home/data/CY/Datasets/Prostate_HCRUDB/source_like')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)

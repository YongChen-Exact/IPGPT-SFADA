import argparse
import os
import h5py
import SimpleITK as sitk
import numpy as np
import torch
from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from utils.metrics import hd95_fast
from networks.net_factory import net_factory
from networks.progressive_teacher import ProT

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/home/data/CY/Datasets/Prostate_HCRUDB",
                    help="Path to the target dataset root directory")
parser.add_argument('--model', type=str, default='unet', help='data_name')
parser.add_argument("--checkpoint", type=str, default="/home/data/CY/code/IPGPT-SFADA/Model/RUNMC_BMC_to_HCRUDB/unet_best_model.pth",
                    help='Path to the target model checkpoint')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# generate_masks
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    hd95 = hd95_fast(pred, gt, (3, 0.5, 0.5))
    return dice, hd95


def test_single_volume_fast(case, net, classes, args, patch_size=[256, 256], batch_size=24):
    h5f = h5py.File(args.root_path + "/test_set/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f["label"][:]
    spacing = h5f["voxel_spacing"]
    prediction = np.zeros_like(label)
    ind_x = np.array([i for i in range(image.shape[0])])
    for ind in ind_x[::batch_size]:
        if ind + batch_size < image.shape[0]:
            stacked_slices = image[ind:ind + batch_size, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]
            zoomed_slices = zoom(stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                _, out = net(input)
                out = torch.argmax(torch.softmax(out, dim=1), dim=1)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind:ind + batch_size, ...] = pred
        else:
            stacked_slices = image[ind:, ...]
            z, x, y = stacked_slices.shape[0], stacked_slices.shape[1], stacked_slices.shape[2]
            zoomed_slices = zoom(stacked_slices, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(zoomed_slices).unsqueeze(1).float().cuda()
            net.eval()
            with torch.no_grad():
                _, out = net(input)
                out = torch.argmax(torch.softmax(out, dim=1), dim=1)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind:, ...] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    path = os.path.join(os.path.dirname(args.checkpoint), 'predictions')
    if not os.path.exists(path):
        os.makedirs(path)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, path +'/'+ case.replace(".h5", "") + "_pred.nii")
    return np.array(metric_list)


def Inference(args):
    image_list = sorted(os.listdir(args.root_path + "/test_set"))
    stu_model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    tch_model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    model = ProT(stu_model, tch_model, 0.98).cuda()
    model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    print("init weight from {}".format(args.checkpoint))
    model.eval()
    segmentation_performance = []
    for case in tqdm(image_list):
        metric = test_single_volume_fast(case, model, args.num_classes, args)
        segmentation_performance.append(metric)
    segmentation_performance = np.array(segmentation_performance)
    return segmentation_performance.mean(axis=0), segmentation_performance.std(axis=0)


if __name__ == '__main__':
    args = parser.parse_args()
    metric = Inference(args)
    print(f"Dice  = {metric[0][0][0]:.4f} ± {metric[1][0][0]:.4f}")
    print(f"HD95  = {metric[0][0][1]:.4f} ± {metric[1][0][1]:.4f}")

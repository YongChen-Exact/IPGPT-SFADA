import argparse
import os
import h5py
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory
from networks.progressive_teacher import ProT
from utils.metrics import hd95_fast, asd_fast

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="/home/data/CY/Datasets/MMS/Heart_B", help="Name of Experiment")
parser.add_argument('--model', type=str,
                    default='unet', help='data_name')
parser.add_argument("--model_path", type=str,
                    default="/home/data/CY/codes/IPGPT-SFADA/Model/Heart_A_to_Heart_B/")
parser.add_argument("--target", type=str, default="Heart_A_to_B", help="experiment_data name")
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# generate evaluation metrics
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    hd95 = hd95_fast(pred, gt, (3, 0.5, 0.5))
    asd = asd_fast(pred, gt, (3, 0.5, 0.5))
    return dice, hd95, asd


def t_single_volume_fast(case, net, classes, FLAGS, patch_size=[256, 256], batch_size=24):
    h5f = h5py.File(FLAGS.root_path + "/test_set/{}".format(case), 'r')
    image = h5f['image'][:]
    label = h5f["label"][:]
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
    return np.array(metric_list)


def Inference(FLAGS):
    image_list = sorted(os.listdir(FLAGS.root_path + "/test_set"))
    files = os.listdir(FLAGS.model_path)
    pth_files = [file for file in files if file.endswith(".pth")]
    sorted_files = sorted(pth_files)
    with open(FLAGS.model_path + 'output_{}.txt'.format(FLAGS.target), 'w') as file:
        for file1 in sorted_files:
            snapshot_path = FLAGS.model_path + file1
            src_model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
            ema_model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
            model = ProT(src_model, ema_model, 0.98).cuda()
            model.load_state_dict(torch.load(snapshot_path)["state_dict"])
            print("init weight from {}".format(snapshot_path))
            model.eval()
            metrics = []
            for case in tqdm(image_list):
                metric = t_single_volume_fast(case, model, FLAGS.num_classes, FLAGS)
                metrics.append(metric)
            metrics = np.array(metrics)
            file.write("model_name = " + snapshot_path + "\n")
            file.write("---------------------dice-------------------" + "\n")
            file.write("LV dice = mean-sd = " + str(metrics.mean(axis=0)[0][0]) + "-" + str(
                metrics.std(axis=0)[0][0]) + "\n")
            file.write("MYO dice = mean-sd = " + str(metrics.mean(axis=0)[1][0]) + "-" + str(
                metrics.std(axis=0)[1][0]) + "\n")
            file.write("RV dice = mean-sd = " + str(metrics.mean(axis=0)[2][0]) + "-" + str(
                metrics.std(axis=0)[2][0]) + "\n")
            file.write("mean dice = mean-sd = " + str(metrics.mean(axis=0).mean(axis=0)[0]) + "-" + str(
                metrics.std(axis=0).mean(axis=0)[0]) + "\n")
            file.write("-----------------------hd95--------------------" + "\n")
            file.write("LV hd95 = mean-sd = " + str(metrics.mean(axis=0)[0][1]) + "-" + str(
                metrics.std(axis=0)[0][1]) + "\n")
            file.write("MYO hd95 = mean-sd = " + str(metrics.mean(axis=0)[1][1]) + "-" + str(
                metrics.std(axis=0)[1][1]) + "\n")
            file.write("RV hd95 = mean-sd = " + str(metrics.mean(axis=0)[2][1]) + "-" + str(
                metrics.std(axis=0)[2][1]) + "\n")
            file.write("mean hd95 = mean-sd = " + str(metrics.mean(axis=0).mean(axis=0)[1]) + "-" + str(
                metrics.std(axis=0).mean(axis=0)[1]) + "\n")
            file.write("-----------------------asd--------------------" + "\n")
            file.write("LV asd = mean-sd = " + str(metrics.mean(axis=0)[0][2]) + "-" + str(
                metrics.std(axis=0)[0][2]) + "\n")
            file.write("MYO asd = mean-sd = " + str(metrics.mean(axis=0)[1][2]) + "-" + str(
                metrics.std(axis=0)[1][2]) + "\n")
            file.write("RV asd = mean-sd = " + str(metrics.mean(axis=0)[2][2]) + "-" + str(
                metrics.std(axis=0)[2][2]) + "\n")
            file.write("mean asd = mean-sd = " + str(metrics.mean(axis=0).mean(axis=0)[2]) + "-" + str(
                metrics.std(axis=0).mean(axis=0)[2]) + "\n")
            file.write("\n")


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(FLAGS.root_path)
    print("dice, hd95, asd    (mean-std)")
    print(metric)

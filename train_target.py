import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.dataset import BaseDataSets, active_DataSets, unlabeled_DataSets, RandomGenerator
from networks.net_factory import net_factory
from networks.progressive_teacher import ProT
from utils import ramps, util, losses
from val_2D import test_single_volume_fast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/home/data/CY/Datasets/Prostate_HCRUDB",
                    help="Path to the target dataset")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--checkpoint", type=str,
                    default="/home/data/CY/code/IPGPT-SFADA/Model/Prostate_RUNMC_BMC_unet/unet_best_model.pth",
                    help='Path to the pre-trained source model checkpoint')
parser.add_argument("--ip_path", type=str,
                    default="/home/data/CY/code/IPGPT-SFADA/Results/selection_list/Prostate_HCRUDB.txt",
                    help="Path to the selected influential points list (txt file)")
parser.add_argument("--source", type=str, default="RUNMC_BMC", help="Name of the source domain dataset")
parser.add_argument("--target", type=str, default="HCRUDB", help="Name of the target domain dataset")
parser.add_argument("--epochs", type=int, default=100, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size per gpu")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.03,  # 0.03 for SGD   0.0001 for adam
                    help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=2025, help="random seed")
args = parser.parse_args()


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_dataset = active_DataSets(base_dir=args.root_path, split="train",
                                      transform=RandomGenerator(args.patch_size), active_path=args.ip_path)
    unlabeled_dataset = unlabeled_DataSets(base_dir=args.root_path, transform=RandomGenerator(args.patch_size),
                                           active_path=args.ip_path)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    stu_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    tch_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    stu_model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    tch_model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    model = ProT(stu_model, tch_model, 0.98).cuda()
    iter_num = 0
    start_epoch = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False,
                                worker_init_fn=worker_init_fn)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=False, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    model.train()
    ce_loss = CrossEntropyLoss()
    writer = SummaryWriter(snapshot_path + "/log")
    max_iterations = (len(labeled_loader) + len(unlabeled_loader)) * epochs
    best_performance = 0.6
    iter_num = int(iter_num)
    iterator = tqdm(range(start_epoch, epochs), ncols=60)
    mc_iteration = 10
    loss_coef = 1
    for epoch_num in iterator:
        all_pseudo_labels = []
        all_un_labels = []
        uncer_ths = []
        for i_batch, (labeled_data, unlabeled_data) in enumerate(zip(labeled_loader, unlabeled_loader)):
            image_batch, label_batch, name_batch = (labeled_data["image"], labeled_data["label"], labeled_data["name"])
            image_batch, label_batch = (image_batch.cuda(), label_batch.cuda())
            un_image_batch, un_label_batch, un_name_batch = (
                unlabeled_data["image"], unlabeled_data["label"], unlabeled_data["name"])
            un_image_batch, un_label_batch = (un_image_batch.cuda(), un_label_batch.cuda())
            _, outputs = model(image_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            supervised_loss1 = 0.5 * (ce_loss(outputs, label_batch.long()) + losses.dice_loss(outputs_soft[:, 1, ...],label_batch))
            outputs_tch = []
            model.tch_model.train()
            for _ in range(mc_iteration):
                with torch.no_grad():
                    _, output = model.tch_model(un_image_batch)
                    outputs_tch.append(output)
            all_predictions = torch.stack(outputs_tch)
            foreground_probs = torch.nn.functional.softmax(torch.squeeze(all_predictions[:, 1, ...]), dim=1)
            conf_th = foreground_probs.max(1)[0].mean()
            std_dev = torch.std(all_predictions, dim=0)
            uncer_th = torch.mean(std_dev)
            outputs_tch = torch.stack(outputs_tch).mean(0)
            probs_w, pseudo_labels = torch.nn.functional.softmax(outputs_tch, dim=1).max(1)
            all_pseudo_labels.append(pseudo_labels.cpu())
            all_un_labels.append(un_label_batch.cpu())
            _, un_output = model(un_image_batch)
            un_outputs_soft = torch.softmax(un_output, dim=1)
            supervised_loss2 = 0.5 * (
                        ce_loss(un_output, pseudo_labels.long()) + losses.dice_loss(un_outputs_soft[:, 1, ...], pseudo_labels))
            difficulty_score = uncer_th / conf_th
            uncer_ths.append(difficulty_score.cpu())
            loss_coef *= (1 - 0.005 * torch.exp(-1 / difficulty_score))
            loss = loss_coef * supervised_loss1 + (1 - loss_coef) * supervised_loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
            writer.add_scalar("lr", lr_, iter_num)
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = np.zeros(3)
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume_fast(sampled_batch["image"], sampled_batch["label"], model,
                                                           classes=num_classes, patch_size=args.patch_size)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar("info/model_val_{}_dice".format(class_i + 1), metric_list[class_i], iter_num)
                performance = np.mean(metric_list)
                writer.add_scalar("info/model_val_mean_dice", performance, iter_num)
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, "model_iter_{}_dice_{}.pth".format(
                        iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(epoch_num, model, optimizer, loss, save_best)
                logging.info("iteration %d : model_mean_dice: %f" % (iter_num, performance))
            model.train()
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, "model_iter_" + str(iter_num) + ".pth")
                util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                break
        all_pseudo_labels = torch.cat(all_pseudo_labels, dim=0)
        all_un_labels = torch.cat(all_un_labels, dim=0)
        pseudo_labels_binary = (all_pseudo_labels > 0).float()
        un_labels_binary = (all_un_labels > 0).float()
        dice_score = dice_coefficient(pseudo_labels_binary, un_labels_binary)
        with open(snapshot_path + "/pseudo_dice.txt", "a") as f:
            f.write(f"{epoch_num}:{dice_score:.4f}\n")
        with open(snapshot_path + "/uncer_th.txt", "a") as f:
            f.write(f"{epoch_num}:{np.mean(uncer_ths):.4f}\n")
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    snapshot_path = "./Model/{}_to_{}".format(args.source, args.target)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

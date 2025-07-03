from networks.deeplab import Deeplab
from networks.enet import ENet
from networks.nnunet import initialize_network
from networks.padl import UNet_PADL
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_Feat


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_feat":
        net = UNet_Feat(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_padl":
        net = UNet_PADL(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "deeplab":
        net = Deeplab(num_classes=1, pretrained=False, inversion=True).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net

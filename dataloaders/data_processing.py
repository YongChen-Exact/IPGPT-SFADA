import argparse
import h5py
import numpy as np
import SimpleITK as sitk
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/home/data/CY/Datasets",
                    help="Path to the dataset root directory")
parser.add_argument("--domain", type=str, default="Prostate_RUNMC_BMC",
                    help="Name of the specific dataset/domain to use")
args = parser.parse_args()

all_cases = os.listdir(args.root_path+r"/{}/imagesTr".format(args.domain))
random.shuffle(all_cases)
training_set = random.sample(all_cases, int(len(all_cases) * 0.7))
val_testing_set = [i for i in all_cases if i not in training_set]
val_set = random.sample(val_testing_set, round(len(all_cases) * 0.1))
test_set = [i for i in val_testing_set if i not in val_set]


class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())



slice_num = 0
for case in training_set:
    image = args.root_path+r"/{}/imagesTr/{}".format(args.domain, case)
    label = args.root_path+r"/{}/labelsTr/{}".format(args.domain, case)
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)
    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)
    image_array_recorrected = MedicalImageDeal(image_array, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()
    out_dir = os.path.join(args.root_path, args.domain, "training_set")
    os.makedirs(out_dir, exist_ok=True)
    for slice_ind in range(image_array.shape[0]):
        out_path = os.path.join(out_dir,"{}_slice_{}.h5".format(case.replace(".nii", "").replace(".nii.gz", ""), slice_ind))
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('image', data=image_array_recorrected_norm[slice_ind], compression="gzip")
            f.create_dataset('label', data=label_array[slice_ind], compression="gzip")
        slice_num += 1

val_num = 0
for case in val_set:
    image = args.root_path+r"/{}/imagesTr/{}".format(args.domain, case)
    label = args.root_path+r"/{}/labelsTr/{}".format(args.domain, case)
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)
    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)
    image_array_recorrected = MedicalImageDeal(image_array, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()
    out_dir = os.path.join(args.root_path, args.domain, "val_set")
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, case.replace(".nii", "").replace(".nii.gz", "")+'.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('image', data=image_array_recorrected_norm, compression="gzip")
        f.create_dataset('label', data=label_array, compression="gzip")
        f.create_dataset('voxel_spacing', data=image_itk.GetSpacing(), compression="gzip")
        f.close()
    val_num += 1

test_num = 0
for case in test_set:
    image = args.root_path+r"/{}/imagesTr/{}".format(args.domain, case)
    label = args.root_path+r"/{}/labelsTr/{}".format(args.domain, case)
    image_itk = sitk.ReadImage(image)
    label_itk = sitk.ReadImage(label)
    image_array = sitk.GetArrayFromImage(image_itk)
    label_array = sitk.GetArrayFromImage(label_itk)
    image_array_recorrected = MedicalImageDeal(image_array, percent=0.99).valid_img
    image_array_recorrected_norm = (image_array_recorrected-image_array_recorrected.mean()) / image_array_recorrected.std()
    out_dir = os.path.join(args.root_path, args.domain, "test_set")
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, case.replace(".nii", "").replace(".nii.gz", "")+'.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('image', data=image_array_recorrected_norm, compression="gzip")
        f.create_dataset('label', data=label_array, compression="gzip")
        f.create_dataset('voxel_spacing', data=image_itk.GetSpacing(), compression="gzip")
        f.close()
    test_num += 1
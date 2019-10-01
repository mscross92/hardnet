import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys
import json


types = ['0','1','2','3','4','5','6','7','8','9','10'] # included images
n_patches = 3408


def mean_image(patches):
    mean = np.mean(patches)
    return mean

def std_image(patches):
    std = np.std(patches)
    return std

class TURBID(data.Dataset):

    def __init__(self, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir,val_set):
        """Return a Tensor containing the patches
        """
        imgs = []
        labels = []
        counter = 0
        for nn in types:
            
            sequence_path = os.path.join(data_dir, '/',str(nn),'/',str(ii))+'.png'
            patch = cv2.imread(sequence_path, 0)
            patch = cv2.resize(patch, (32, 32))
            patch = np.array(patch, dtype=np.uint8)
            patches.append(patch)
            labels.append(ii)
            counter += 1
        print(counter)
        return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels)

if __name__ == '__main__':
    # need to be specified
    try:
        path_to_patches_dir = sys.argv[1]
        output_dir  = sys.argv[3]
        val_set_idx  = sys.argv[4]
    except:
        print("Wrong input format. Try python HPatchesDatasetCreator.py path_to_hpatches path_to_splits_json output_dir")
        sys.exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    t = "train"
    hPatches = HPatches()
    split = 'a'
    images, labels = hPatches.read_image_file(path_to_patches_dir,val_set_idx)
    with open(os.path.join(output_dir, 'hpatches_split_' + split +  '_' + t + '.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print(split, t, 'Saved')


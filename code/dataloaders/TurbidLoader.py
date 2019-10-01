import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys
import json


types = ['0','1','2','3','4','5','6','7','8','9','10'] # included images

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

    def read_image_file(self, data_dir):
        """Return a Tensor containing the patches
        """
        imgs = []
        labels = []
        counter = 0
        for nn in types:
            sequence_path = os.path.join(data_dir, '/',str(nn),'.jpg')
            gray = cv2.imread(sequence_path)
            # convert to grayscale
            lin_img = ((gray/255 + 0.055) / 1.055) ** 2.4
            gray_lin = 0.212*lin_img[:,:,0] + 0.7152*lin_img[:,:,1] + 0.0722*lin_img[:,:,2]
            gray = 1.055 * (gray_lin**(1/2.4)) - 0.055
            gray = (gray*255).astype('uint8')
            gray = np.array(gray, dtype=np.uint8)
            imgs.append(gray)
            labels.append(nn)
            counter += 1
        print(counter,'images loaded')
        return torch.ByteTensor(np.array(imgs, dtype=np.uint8)), torch.LongTensor(labels)

if __name__ == '__main__':
    # need to be specified
    try:
        path_to_imgs_dir = sys.argv[1]
        output_dir  = sys.argv[2]
    except:
        print("Wrong input format. Try python HPatchesDatasetCreator.py path_to_hpatches path_to_splits_json output_dir")
        sys.exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    t = "train"
    trbd = TURBID()
    images, labels = trbd.read_image_file(path_to_imgs_dir)
    with open(os.path.join(output_dir, 'turbid_imgs.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print(t, 'images saved')


import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys
import json
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=5, low=-15, upp=15):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

types = [2,3,4,5,6,7,8,9,10,11,12] # included images
rot_dists = get_truncated_normal().rvs(25)

# def mean_image(patches):
#     mean = np.mean(patches)
#     return mean

# def std_image(patches):
#     std = np.std(patches)
#     return std

class TURBID(data.Dataset):

    def __init__(self, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir, val_set_idx):
        # """Return a Tensor containing the images
        # """
        # imgs = []
        # labels = []
        # counter = 0
        # for nn in types:
        #     pth = data_dir + '/' + str(nn) +'.jpg'
        #     gray = cv2.imread(pth)
        #     # convert to grayscale
        #     lin_img = ((gray/255 + 0.055) / 1.055) ** 2.4
        #     gray_lin = 0.212*lin_img[:,:,0] + 0.7152*lin_img[:,:,1] + 0.0722*lin_img[:,:,2]
        #     gray = 1.055 * (gray_lin**(1/2.4)) - 0.055
        #     gray = (gray*255).astype('uint8')
        #     gray = np.array(gray, dtype=np.uint8)
        #     imgs.append(gray)
        #     labels.append(nn-2)
        #     counter += 1
        # print(counter,'images loaded')
        # return torch.ByteTensor(np.array(imgs, dtype=np.uint8)), torch.LongTensor(labels)
        """Reads images and extracts training patches, performing offline augmentation
        """
        # get feature points
        fps_str = data_dir + '/features.txt'
        fps = []
        lines = [line.strip() for line in open(fps_str)]
        for line in lines:
            list = line.split(',')
            kp = cv2.KeyPoint(x=float(list[0]), y=float(list[1]), _size=float(list[2]), _angle=float(list[3]),
                            _response=float(list[4]), _octave=int(list[5]), _class_id=int(list[6]))
            if kp.size<12:
                kp.size = 12
            fps.append(kp)
        del list

        # get validation set
        val_idxs = np.loadtxt(data_dir + '/validation_location_idxs_set'+str(val_set_idx)+'.txt',delimiter=',')
        
        ptchs = []
        labels = []
        counter = 0
        for nn in types: # iterate through every image
            # load image
            pth = data_dir + '/' + str(nn) +'.jpg'
            print(pth)
            gray = cv2.imread(pth)
            # convert to grayscale
            lin_img = ((gray/255 + 0.055) / 1.055) ** 2.4
            gray_lin = 0.212*lin_img[:,:,0] + 0.7152*lin_img[:,:,1] + 0.0722*lin_img[:,:,2]
            gray = 1.055 * (gray_lin**(1/2.4)) - 0.055
            gray = (gray*255).astype('uint8')
            gray = np.array(gray, dtype=np.uint8)
            (h,w) = gray.shape

            for ll,p in enumerate(fps): # iterate through every point
                if ll not in val_idxs: # check to be included in train set
                    # extract feature details
                    (y,x) = p.pt
                    s = p.size
                    if s<12:
                        s = 12

                    # original patch
                    ptch = gray[int(x-0.5*s):int(x-0.5*s)+int(s),int(y-0.5*s):int(y-0.5*s)+int(s)]
                    ptchs.append(torch.ByteTensor(np.array(ptch, dtype=np.uint8)).cuda())
                    labels.append(ll)

                    # rotated patches
                    rot_dist = get_truncated_normal().rvs(4) # sample angles from normal distribution
                    for r in rot_dist:
                        M = cv2.getRotationMatrix2D((y,x), r, 1.0) # rotate about patch center
                        rotated = cv2.warpAffine(gray, M, (w, h))
                        ptch = rotated[int(x-0.5*s):int(x-0.5*s)+int(s),int(y-0.5*s):int(y-0.5*s)+int(s)]
                        ptchs.append(torch.ByteTensor(np.array(ptch, dtype=np.uint8)).cuda())
                        labels.append(ll)

                    # perspective transform patch
                    

        print(len(ptchs),'patches created from',ll,'locations and',nn,'images')
        return torch.ByteTensor(np.array(ptchs, dtype=np.uint8)), torch.LongTensor(labels)


if __name__ == '__main__':
    # need to be specified
    try:
        path_to_imgs_dir = sys.argv[1]
        output_dir  = sys.argv[2]
        val_set  = sys.argv[3]
    except:
        print("Wrong input format. Try python HPatchesDatasetCreator.py path_to_hpatches path_to_splits_json output_dir")
        sys.exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    t = "train"
    trbd = TURBID()
    images, labels = trbd.read_image_file(path_to_imgs_dir, val_set)
    with open(os.path.join(output_dir, 'turbid_imgs.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print(t, 'images saved')


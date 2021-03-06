# import os
# import numpy as np
# import torch
# import torch.utils.data as data
# import cv2
# import sys
# import json

# # types = ['0','1','2','3','4','5','6','7','8','9']

# types = ['0','1','2','3','4','5','6','7','8','9','10']

# # types = ['0','1','2']
# splits = ['a']
# # splits = ['b']

# #images_to_exclude = ['v_adam', 'v_boat', 'v_graffiti', 'v_there','i_dome']

# def mean_image(patches):
#     mean = np.mean(patches)
#     return mean

# def std_image(patches):
#     std = np.std(patches)
#     return std

# class HPatches(data.Dataset):

#     def __init__(self, train=True, transform=None, download=False, good_fnames = []):
#         self.train = train
#         self.transform = transform

#     def read_image_file(self, data_dir):
#         """Return a Tensor containing the patches
#         """
#         patches = []
#         labels = []
#         counter = 0
#         hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
#         for directory in hpatches_sequences:
#            if (directory in good_fnames):
#             print(directory)
#             for type in types:
#                 sequence_path = os.path.join(data_dir, directory,type)+'.png'
#                 image = cv2.imread(sequence_path, 0)
#                 h, w = image.shape
#                 n_patches = int(h / w)
#                 for i in range(n_patches):
#                     patch = image[i * (w): (i + 1) * (w), 0:w]
#                     patch = cv2.resize(patch, (32, 32))
#                     patch = np.array(patch, dtype=np.uint8)
#                     patches.append(patch)
#                     labels.append(i+counter)
#             counter += n_patches
#         print(counter)
#         return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels)

#     def read_image_file_test(self, data_dir):
#         """Return a Tensor containing the patches
#         """
#         typs = ['0','1','2','3','4','5','6','7','8','9','10','11','12']

#         patches = []
#         labels = []
#         counter = 0
#         hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
#         for directory in hpatches_sequences:
#            if (directory in good_fnames):
#             print(directory)
#             for type in typs:
#                 sequence_path = os.path.join(data_dir, directory,type)+'.png'
#                 image = cv2.imread(sequence_path, 0)
#                 h, w = image.shape
#                 n_patches = int(h / w)
#                 for i in range(n_patches):
#                     patch = image[i * (w): (i + 1) * (w), 0:w]
#                     patch = cv2.resize(patch, (32, 32))
#                     patch = np.array(patch, dtype=np.uint8)
#                     patches.append(patch)
#                     labels.append(i+counter)
#             counter += n_patches
#         print(counter)
#         return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels)

# if __name__ == '__main__':
#     # need to be specified
#     try:
#         path_to_hpatches_dir = sys.argv[1]
#         path_to_splits_json = sys.argv[2]
#         output_dir  = sys.argv[3]
#     except:
#         print("Wrong input format. Try python HPatchesDatasetCreator.py path_to_hpatches path_to_splits_json output_dir")
#         sys.exit(1)
#     splits_json = json.load(open(path_to_splits_json, 'rb'))
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for split in splits:
#         if len(split) == 1:
#             t = 'train'
#         else:
#             t = 'test'# view and illum are kind of train/test for each other

#         good_fnames = splits_json[split][t]
#         hPatches = HPatches(good_fnames = good_fnames)
#         images, labels = hPatches.read_image_file(path_to_hpatches_dir)
#         with open(os.path.join(output_dir, 'hpatches_split_' + split +  '_' + t + '.pt'), 'wb') as f:
#             torch.save((images, labels), f)
#         print(split, t, 'Saved')

#         t = 'test'
#         good_fnames = splits_json[split][t]
#         hPatches = HPatches(good_fnames = good_fnames)
#         images, labels = hPatches.read_image_file_test(path_to_hpatches_dir)
#         with open(os.path.join(output_dir, 'hpatches_split_' + split +  '_' + t + '.pt'), 'wb') as f:
#             torch.save((images, labels), f)
#         print(split, t, 'Saved')


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

class HPatches(data.Dataset):

    def __init__(self, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir,val_set):
        """Return a Tensor containing the patches
        """
        # patches = []
        # labels = []
        # counter = 0
        # hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        # for directory in hpatches_sequences:
        #     print(directory)
        #     for type in types:
        #         sequence_path = os.path.join(data_dir, directory,type)+'.png'
        #         image = cv2.imread(sequence_path, 0)
        #         h, w = image.shape
        #         n_patches = int(h / w)
        #         for i in range(n_patches):
        #             patch = image[i * (w): (i + 1) * (w), 0:w]
        #             patch = cv2.resize(patch, (32, 32))
        #             patch = np.array(patch, dtype=np.uint8)
        #             patches.append(patch)
        #             labels.append(i+counter)
        #     counter += n_patches
        # print(counter)
        val_p = np.loadtxt('/content/hardnet/data/validation_location_idxs_set'+str(val_set)+'.txt',delimiter=',')

        patches = []
        labels = []
        counter = 0
        for nn in types:
            for ii in range(n_patches):
                if ii not in val_p:
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


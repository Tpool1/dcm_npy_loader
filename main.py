import torchio
import os
import numpy as np
import pydicom as dicom
import time
import torch
import random
import math

prev_time = time.time()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

imgs_shape = (512, 512)

load_dir = '/scratch/Duke-Breast-Cancer-MRI_v120201203/Duke-Breast-Cancer-MRI'

load_paths = list()
for (dirpath, dirnames, filenames) in os.walk(load_dir):
    load_paths += [os.path.join(dirpath, file) for file in filenames]

# get random subset of list
percent_sample = 50
total_imgs = len(load_paths) * (percent_sample*0.01)

# round down
total_imgs = math.floor(total_imgs)

new_path_list = []
i = 0
while i < total_imgs:
    rand_choice = random.choice(load_paths)

    if rand_choice not in new_path_list:
        new_path_list.append(rand_choice)
        i = i + 1

load_paths = new_path_list

img_list = []
for path in load_paths:

    try: 
        img = dicom.dcmread(path)
        img_shape = img.pixel_array.shape
        id = img.PatientID

        if img_shape == imgs_shape:
            for c in id:
                if not c.isdigit():
                    id = id.replace(c, '')

            subject_dict = {
                'one image': torchio.ScalarImage(path),
                'id': id
            }

            subject = torchio.Subject(subject_dict)

            img_list.append(subject)

    except:
        print('image ' + str(path) + ' could not be loaded')

dataset = torchio.SubjectsDataset(img_list, load_getitem=True)
print('Total length of dataset: ' + str(len(dataset)))

device = torch.device("cpu")

img_array = np.empty(shape=(len(dataset), (imgs_shape[0]*imgs_shape[1])+1))

loader = torch.utils.data.DataLoader(dataset)
for i in range(len(dataset)):
    id = torch.tensor([int(next(iter(loader))['id'][0])])
    images = next(iter(loader))['one image']['data']

    images = images.to(device, torch.uint8)

    i = 0
    for image in images:

        images = images.cpu().numpy()

        images = images.flatten()

        images = np.append(images, id)

        img_array[i] = images

        i = i + 1

np.save('converted_imgs/img_array.npy', img_array)

after_time = time.time()

load_time = after_time - prev_time
print(load_time)

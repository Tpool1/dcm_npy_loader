import torchio
import os
import numpy as np
import pydicom as dicom
import time
import torch

prev_time = time.time()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

load_dir = '/scratch/Duke-Breast-Cancer-MRI_v120201203/Duke-Breast-Cancer-MRI'

load_paths = list()
for (dirpath, dirnames, filenames) in os.walk(load_dir):
    load_paths += [os.path.join(dirpath, file) for file in filenames]

img_list = []
for path in load_paths:

    try: 
        img = dicom.dcmread(path)
        id = img.PatientID

        for c in id:
            if not c.isdigit():
                id = id.replace(c, '')

        subject_dict = {
            'one image': torchio.ScalarImage(path),
            'id': id
        }

        subject = torchio.Subject(subject_dict)

        img_list.append(subject)

        dataset = torchio.SubjectsDataset(img_list, load_getitem=True)
    except:
        print('image ' + str(path) + ' could not be loaded')

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

img_tensor = torch.empty(size=(len(dataset), 512, 512, 1))

for i in range(len(dataset)):
    loader = torch.utils.data.DataLoader(dataset)
    id = next(iter(loader))['id']
    images = next(iter(loader))['one image']['data']

    images = images.to(device, torch.uint8)

    j = 0
    for image in images:
        if tuple(image.shape) != (1, 512, 512, 1):

            # remove image if not correct shape
            images = torch.cat([images[0:j], images[j+1:]])

        j = j + 1

    img_tensor[i] = images

img_tensor = img_tensor.to(device, torch.uint8)

img_tensor = img_tensor.cpu().numpy()

np.save('converted_imgs/img_array.npy', img_tensor)

after_time = time.time()

load_time = after_time - prev_time
print(load_time)

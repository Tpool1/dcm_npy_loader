import torchio
import os
import numpy as np
import pydicom
import time
import torch
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

def dcm_npy_loader(load_dir, shape=(512, 512), load=True):
    if not load:
        prev_time = time.time()
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

        load_paths = list()
        for (dirpath, dirnames, filenames) in os.walk(load_dir):
            load_paths += [os.path.join(dirpath, file) for file in filenames]

        # get random subset of list
        percent_sample = 100
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

        files = []
        ids = []
        for path in load_paths:
            try:
                file = pydicom.dcmread(path)
                if file.pixel_array.shape == shape:
                    files.append(file)
                    id = file.PatientID
                    ids.append(id)
            except:
                print("Image " + path + " could not be loaded")

        # remove duplicates
        ids = list(set(ids))

        all_img3d = []
        all_ids = []
        all_sliceLocs = []
        for id in ids:
        
            # skip files with no SliceLocation
            slices = []
            skipcount = 0
            for f in files:
                if f.PatientID == id:
                    if hasattr(f, 'SliceLocation'):
                        slices.append(f)
                    else:
                        skipcount = skipcount + 1
                
            # ensure slices are in the correct order
            slices = sorted(slices, key=lambda s: s.SliceLocation)

            # pixel aspects, assuming all slices are the same
            ps = slices[0].PixelSpacing
            ss = slices[0].SliceThickness
            ax_aspect = ps[1]/ps[0]
            sag_aspect = ps[1]/ss
            cor_aspect = ss/ps[0]

            # create 3D array
            img_shape = list(slices[0].pixel_array.shape)
            img_shape.append(len(slices))
            img3d = np.zeros(img_shape, dtype=np.int8)

            p_id = slices[0].PatientID
            # get only numbers from patient id
            p_id = [int(s) for s in p_id if s.isdigit()]
            p_id = int(''.join([str(i) for i in p_id]))

            slice_locs = []

            # fill 3D array with the images from the files
            for i, s in enumerate(slices):
                img2d = s.pixel_array

                slice_locs.append(s.get('SliceLocation'))
                if list(img2d.shape) == img_shape[:2]:
                    img3d[:, :, i] = img2d
            
            all_ids.append(p_id)
            all_img3d.append(img3d)
            all_sliceLocs.append(slice_locs)

        after_time = time.time()
        load_time = after_time - prev_time
        print(load_time)

        data = [all_img3d, all_ids, all_sliceLocs]

        with open('data\\Duke-Breast-Cancer-MRI\\data', 'wb') as fp:
            pickle.dump(data, fp)

    else:
        with open('data\\Duke-Breast-Cancer-MRI\\data', 'rb') as fp:
            data = pickle.load(fp)

    return data

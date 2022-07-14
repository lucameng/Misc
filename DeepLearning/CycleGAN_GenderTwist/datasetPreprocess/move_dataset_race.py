# -*- coding: utf-8 -*-
from imageio import imread, imsave
import cv2
import glob, os
from tqdm import tqdm
import pandas as pd


WIDTH = 128
HEIGHT = 128
COUNT = 6666


data_dir = f'dataset_asian'
male_dir = f'dataset_asian/male'
female_dir = f'dataset_asian/female'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(male_dir):
    os.mkdir(male_dir)
if not os.path.exists(female_dir):
    os.mkdir(female_dir)


male_count = 0
female_count = 0

train_df = pd.read_csv('./fairface_label_val.csv')


def read_process_save(read_path, save_path):
    image = imread(read_path)
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]    
    image = cv2.resize(image, (WIDTH, HEIGHT))
    imsave(save_path, image)

target1 = 'gender'
target2 = 'race'



i = 0
length = len(train_df)
while i < length:
    if train_df.iloc[i, 3] == ('East Asian' or 'Southeast Asian'):
        fname = train_df.iloc[i, 0]
        if train_df.iloc[i, 2] == 'Male':
            read_process_save(os.path.join('fairface', fname), os.path.join(male_dir, fname)) # 男
        elif train_df.iloc[i, 2] == 'Female':
            read_process_save(os.path.join('fairface', fname), os.path.join(female_dir, fname)) # nv
    i += 1


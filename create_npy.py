import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2

# #####################  im 2 npy  #################

path = r'../Dataset/training_data/training_data_bmp/masks/'
out_path = r'../Dataset/training_data/training_data_bmp/masks_npy_in10/'


def get_file(path, rule='.bmp'):
    all = []
    img_list = sorted(os.listdir(path))  # 文件名按字母排序
    img_nums = len(img_list)
    for i in range(img_nums):
        img_name = path + img_list[i]
        all.append(img_name)
    return all


if __name__ == '__main__':
    paths = get_file(path, rule='.bmp')
    imgs10 = np.empty([10, 256, 256], dtype=int)
    i = 0
    for ims in paths:
        file_name = ims.strip(path)
        im1 = cv2.imread(ims)
        im2 = np.squeeze(np.array(im1[:, :, 0]))
        imgs10[i, :, :] = im2
        i += 1
        if i == 9:
            save_path = out_path + file_name + '.npy'
            #print(save_path)
            np.save(save_path, imgs10)
            i = 0
            imgs10 = np.empty([10, 256, 256], dtype=int)

"""
print("------show npy's shape_size---------")
data = np.load('../dataset1/jpgdata/test_jpg/000000050755_bicLRx4.npy')
print(data.shape)
"""
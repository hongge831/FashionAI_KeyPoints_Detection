# -- coding: utf-8 --
import os, sys
# sys.path.append('../utils/')
import torch
import torch.utils.data as data
import numpy as np
import os
import math
from PIL import Image
import cv2
import csv
import Mytransforms
import matplotlib.pyplot as plt

class dataset_loader(data.Dataset):
    #这里的stride是什么？

    def __init__(self, numPoints,img_dir, ann_path, stride, transforms=None, sigma = 15):
        #初始化属性，在类内可用
        self.numPints = numPoints
        self.sigma = sigma#15 #9 #15
        self.stride = stride
        self.img_dir = img_dir
        self.transforms = transforms
        self.anns = []
        self.info = []
        with open(ann_path,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.anns.append(row)
        #第一行存储csv文件的列名等信息
        self.info.append(self.anns[0])
        #后面将不同的服饰的点的信息存成列表
        self.anns=self.anns[1:]


    def __getitem__(self, index):
        #这个函数是将数据对象变成可迭代的形式，迭代的标签是index
        # ---------------- read info -----------------------
        ann = self.anns[index]
        numpoints = self.numPints
        img_path = os.path.join(self.img_dir, ann[0])
        img = cv2.imread(img_path) # BGR
        img_margin = 100 #给截图的图片预留一些空间
        #读取图片的类别信息，但是没什么用
        catergory = ann[1]
        kpt = _get_keypoints(ann,numpoints)

        #########################################################################################
        # 1.获取所有关键点的左上角和右下角的点
        left_x = np.min(kpt[:, 0])
        left_y = np.min(kpt[:, 1])
        right_x = np.max(kpt[:, 0])
        right_y = np.max(kpt[:, 1])
        # 2.直接截取原图片(这里截取的方式是行列截取，正好与坐标轴的XY顺序相反)
        #这里还需要加一句判断，加了margin之后会不会超过原始的0位置了
        pad_up = 40
        pad_left = 40
        if(left_y - img_margin < 0):
            pad_up = left_y
        if (left_x - img_margin < 0):
            pad_left = left_x
        img_crop = img[int(left_y - pad_up):int(right_y + img_margin), int(left_x - pad_left):int(right_x + img_margin)]
        # 3.numpy的批量处理坐标的值获得新的kpt
        kpt[:, 0] = kpt[:, 0] - left_x + pad_left
        kpt[:, 1] = kpt[:, 1] - left_y + pad_up
        #4.更新目标中心点
        # center = [img_crop.shape[0] / 2, img_crop.shape[1] / 2]
        img = img_crop
        ###############################如果要将图片的信息做修改，应该在这里做修改######################

        # ----------------- transform ----------------------
        #物体中心点（可以改进的地方）
        center = [img.shape[0]/2,img.shape[1]/2]
        
        # ----------------- transform ----------------------
        #这里transform以后图片数据集的数量会不会发生变化，是否有一些数据增强的操作？
        if not self.transforms:
            img, kpt = _croppad(img, kpt, center, 384, 384)
        else:
            img, kpt, center = self.transforms(img, kpt, center)
        #---------------------------------------------------
        #生成heatmap的函数，也就是所谓的label或者是groundtruth
        #这里的图片已经被截断成了384*384，这里的kpt经过图像变换之后也一定已经发生变化了
        # cv2.imwrite(os.path.join('e:/test22.jpg'), img)


        heatmaps = _generate_heatmap(img, kpt,self.stride, self.sigma)
        # img2 = cv2.fromarray(heatmaps)
        # cv2.imwrite(os.path.join('e:/test232.bmp'), heatmaps)
        img = np.array(img, dtype=np.float32)
        img -= 128.0
        img /= 255.0

        img = torch.from_numpy(img.transpose((2, 0, 1)))
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)))

        # img = self.trasforms(img)
        # heatmaps = self.trasforms(heatmaps)
        #返回的形式就是图像+label的形式，这里的label就是heatmap
        return img, heatmaps

    def __len__(self):
        return len(self.anns)

def _croppad(img, kpt, center, w, h):
    #如果图片大小小于384怎么办????
    num = len(kpt)
    height, width, _ = img.shape
    new_img = np.empty((h, w, 3), dtype=np.float32)
    new_img.fill(128)

    # calculate offset
    offset_up = -1*(h/2 - center[0])
    offset_left = -1*(w/2 - center[1])

    for i in range(num):
        kpt[i][0] -= offset_left
        kpt[i][1] -= offset_up

    st_x = 0
    ed_x = w
    st_y = 0
    ed_y = h
    or_st_x = offset_left
    or_ed_x = offset_left + w
    or_st_y = offset_up
    or_ed_y = offset_up + h

    if offset_left < 0:
        st_x = -offset_left
        or_st_x = 0
    if offset_left + w > width:
        ed_x = width - offset_left
        or_ed_x = width
    if offset_up < 0:
        st_y = -offset_up
        or_st_y = 0
    if offset_up + h > height:
        ed_y = height - offset_up
        or_ed_y = height
    new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()

    return np.ascontiguousarray(new_img), kpt


def _get_keypoints(ann,numpoints):
    # index_array = [17,18,21,22,23,24,25]
    #根据不同的服饰类型，去读取点的坐标
    if(numpoints==13):
        index_array = [2,3,4,5,6,7,8,11,12,13,14,15,16]
    elif(numpoints==15):
        index_array = [2,3,4,5,6,7,8,9,10,11,12,13,14,19,20]
    elif (numpoints == 14):
        index_array = [2,3,5,6,7,8,9,10,11,12,13,14,15,16]
    elif (numpoints == 4):
        index_array = [17, 18, 19, 20]
    elif (numpoints == 7):
        index_array = [17,18,21,22,23,24,25]

    kpt = np.zeros((numpoints, 3))
    for i in range(numpoints):
        str = ann[index_array[i]]
        [x_str, y_str, vis_str] = str.split('_')
        kpt[i, 0], kpt[i, 1], kpt[i, 2] = int(x_str), int(y_str), int(vis_str)
    return kpt

def _generate_heatmap(img, kpt, stride, sigma):
    #这里的stride是有讲究的是因为8 = 384/48？
    height, width, _ = img.shape
    heatmap = np.zeros((height // stride, width // stride, len(kpt) + 1), dtype=np.float32) # (24 points + background)
    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    for i in range(num):
        if kpt[i][2] == -1:  # not labeled
            continue
        x = kpt[i][0]
        y = kpt[i][1]
        for h in range(height):
            for w in range(width):
                xx = start + w * stride
                yy = start + h * stride
                dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                if dis > 4.6052:
                    continue
                heatmap[h][w][i] += math.exp(-dis)
                if heatmap[h][w][i] > 1:
                    heatmap[h][w][i] = 1

    heatmap[:, :, -1] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)  # for background
    return heatmap

'''
0: labeled but not visble
1: labeled and visble
-1: not labeled

'image_id',
 'image_category',
0'neckline_left',
1'neckline_right',
2 'center_front',
3'shoulder_left',
4 'shoulder_right',
5 'armpit_left',
6 'armpit_right',
7 'waistline_left',
8 'waistline_right',
9 'cuff_left_in',
10 'cuff_left_out',
11 'cuff_right_in',
12 'cuff_right_out',
13 'top_hem_left',
14 'top_hem_right',
15 'waistband_left',
16 'waistband_right',
17 'hemline_left',
18 'hemline_right',
19 'crotch',
20 'bottom_left_in',
21 'bottom_left_out',
22 'bottom_right_in',
23 'bottom_right_out
'''

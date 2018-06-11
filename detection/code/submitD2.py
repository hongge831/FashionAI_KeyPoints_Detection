# -- coding: utf-8 --
import csv
import os
import sys
sys.path.append('../')
sys.path.append('../detection/code/')
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import math, time
import torch
import csv
from utils import util
import datetime
import pandas as pd

def apply_model(oriImg, model, multiplier,numPoints,roi_str):
    stride = 8
    roiPoint = roi_str.split('_')
    newImg = oriImg[int(roiPoint[0]):int(roiPoint[2]), int(roiPoint[1]):int(roiPoint[3])]

    height, width, _ = newImg.shape
    oriImg = newImg
    #height, width, _ = oriImg.shape
    #将图像转化成数组形式，类型是float32
    normed_img = np.array(oriImg, dtype=np.float32)
    #新建一个大小一样的0矩阵（图），另外这个图的通道数就是代预测点的个数
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], numPoints), dtype=np.float32)
    #遍历尺度因数
    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # imgToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, 128)
        #将输入图片补全成标准大小（可能是384*384）
        imgToTest_padded, pad = util.padRightDownCorner(imageToTest, 32, 128)

        input_img = np.transpose(np.float32(imgToTest_padded[:, :, :, np.newaxis]),
                                 (3, 2, 0, 1)) / 255 - 0.5  # required shape (1, c, h, w)

        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())

        # get the features
        # heat1, heat2, heat3, heat4, heat5, heat6 = model(input_var)
        #怎么看是经历了几个阶段的特诊图
        heat = model(input_var)

        # get the heatmap
        heatmap = heat.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0
    thre1 = 0.1
    for part in range(numPoints - 1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # sort by score
    for i in range(numPoints-1):
        all_peaks[i] = sorted(all_peaks[i], key=lambda ele : ele[2],reverse = True)

    keypoints = -1*np.ones((numPoints-1, 3))
    for i in range(numPoints-1):
        if len(all_peaks[i]) == 0:
            continue
        else:
            keypoints[i,0], keypoints[i,1], keypoints[i,2] = all_peaks[i][0][0], all_peaks[i][0][1], 1
    keypoints[:,0] =  keypoints[:,0] + int(roiPoint[1])
    keypoints[:,1] =  keypoints[:,1] + int(roiPoint[0])
    return  keypoints


def write_csv(name, results):
    import csv
    with open(name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)

def prepare_row(ann, keypoints,index_array):
    # cls
    image_name = ann[0]
    category = ann[1]
    keypoints_str = []
    #index_array = [17,18,21,22,23,24,25]
    j=0
    for i in range(24):
        if((i+2) in index_array):
            cell_str = str(int(keypoints[j][0])) + '_' + str(int(keypoints[j][1])) + '_' + str(int(keypoints[j][2]))
            j+=1
        else:
            cell_str = '-1_-1_-1'

        keypoints_str.append(cell_str)
    row = [image_name, category] + keypoints_str
    return row

def read_csv(ann_file):
    info = []
    anns = []
    with open(ann_file, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            anns.append(row)
    info = anns[0]
    anns = anns[1:]
    return info, anns

def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    modelsList = ['../saveparameter/blouse/12000res50.pth.tar','../saveparameter/dress/12000res50.pth.tar','../saveparameter/outwear/12000res50.pth.tar','../saveparameter/skirt/12000res50.pth.tar','../saveparameter/trousers/12000res50.pth.tar']
    annList = ['../data/test/blouse.csv','../data/test/dress.csv','../data/test/outwear.csv','../data/test/skirt.csv','../data/test/trousers.csv']
    saveList = ['../submit/reblouse.csv','../submit/redress.csv','../submit/reoutwear.csv','../submit/reskirt.csv','../submit/retrousers.csv']

    ROIFileList = ['../detection/submit/blouse_test_result.csv', '../detection/submit/dress_test_result.csv', '../detection/submit/outwear_test_result.csv', '../detection/submit/skirt_test_result.csv',
                '../detection/submit/trousers_test_result.csv']

    classNumList = [13,15,14,4,7]
    index_array = [[2,3,4,5,6,7,8,11,12,13,14,15,16],[2,3,4,5,6,7,8,9,10,11,12,13,14,19,20],[2,3,5,6,7,8,9,10,11,12,13,14,15,16],[17,18,19,20],[17,18,21,22,23,24,25]]
    className = ['blouse','dress','outwear','skirt','trousers']
    for i in range(3,5):
        # --------------------------- model -------------------------------------------------------------------------------
        import models.CPM_FPN
        pytorch_model = modelsList[i]
        model = models.CPM_FPN.pose_estimation(class_num=classNumList[i], pretrain=False)
        # -----------------------------------------------------------------------------------------------------------------

        img_dir = '../data/test/'
        ann_path = annList[i]
        result_name = saveList[i]
        # scale_search = [0.5, 0.7, 1.0, 1.3]  # [0.5, 1.0, 1.5]
        scale_search = [0.5, 0.7, 1.0]
        boxsize = 384
        # -------------------------- pytorch model------------------
        state_dict = torch.load(pytorch_model)['state_dict']
        model.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()  # make model in test mode
        # --------------------------------------------------------
        anns = []
        with open(ann_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                anns.append(row)
        ROIanns = []
        with open(ROIFileList[i], 'r') as f:
            reader2 = csv.reader(f)
            for row2 in reader2:
                ROIanns.append(row2)
        #info = anns[0]
        info = ['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front', 'shoulder_left',
                   'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                   'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right',
                   'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in',
                   'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
        anns = anns[1:]
        ROIanns = ROIanns[1:]
        # ---------------------------------------------------------
        num_imgs = len(anns)
        results = []
        #results.append(info)

        for j in range(num_imgs):
            print('{}/{}/{}'.format(i,j, num_imgs))
            ann = anns[j]
            roiann = ROIanns[j]
            image_path = os.path.join(img_dir, ann[0])
            oriImg = cv2.imread(image_path)
            # multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
            multiplier = scale_search
            numPoints = classNumList[i]+1
            keypoints = apply_model(oriImg, model, multiplier,numPoints,roiann[2])
            # cv2.imwrite(os.path.join('./result', ann[0].split('/')[-1]), canvas)
            row = prepare_row(ann, keypoints,index_array[i])
            results.append(row)
        write_csv(result_name, results)
    src = '../submit/'
    colname=['image_id' ,'image_category' ,'neckline_left' ,'neckline_right' ,'center_front' ,'shoulder_left' ,'shoulder_right' ,'armpit_left' ,'armpit_right' ,'waistline_left' ,'waistline_right' ,'cuff_left_in' ,'cuff_left_out' ,'cuff_right_in' ,'cuff_right_out' ,'top_hem_left' ,'top_hem_right' ,'waistband_left' ,'waistband_right' ,'hemline_left' ,'hemline_right' ,'crotch' ,'bottom_left_in' ,'bottom_left_out' ,'bottom_right_in' ,'bottom_right_out']
    result_blouse = pd.read_csv(src + 'reblouse.csv', header=None)
    result_dress= pd.read_csv(src + 'redress.csv', header=None)
    result_outwear = pd.read_csv(src + 'reoutwear.csv', header=None)
    result_skirt = pd.read_csv(src + 'reskirt.csv', header=None)
    result_trousers = pd.read_csv(src + 'retrousers.csv', header=None)
    result_all = pd.concat([result_blouse, result_dress], axis=0)
    result_all = pd.concat([result_all, result_outwear], axis=0)
    result_all = pd.concat([result_all, result_skirt], axis=0)
    result_all = pd.concat([result_all, result_trousers], axis=0)
    result_all.columns = colname
    result_all.to_csv(("../submit/submit_test_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"),index=None)



if __name__ == '__main__':
    main()

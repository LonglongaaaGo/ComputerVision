import cv2 as cv
import argparse
import os
import numpy as np
import math
import scipy.stats as st

def bi_linear_resize(pic, scale):
    """
    bi_linear resize
    :param pic:  img
    :param scale:  scale >0
    :return:  resized img
    """
    h,w,c = pic.shape  # height,width,channel
    th, tw = int(h * scale), int(w * scale)
    # avoid the out of bounds from the original img
    pic = np.pad(pic, ((0, 1), (0, 1), (0, 0)), 'reflect')

    emptyImage = np.zeros((th, tw,c ), np.uint8)
    h_scale = h/th
    w_scale = w/tw
    for i in range(th):
        for j in range(tw):
            # 首先找到在原图中对应的点的(X, Y)坐标
            #first, find the location from the original img
            corr_x = (i + 0.5)*h_scale - 0.5
            corr_y = (j + 0.5)*w_scale - 0.5
            # if i * pic.shape[0] % th == 0 and j * pic.shape[1] % tw == 0:  # 对应的点正好是一个像素点，直接拷贝
            #     #   emptyImage[i, j, k] = pic[int(corr_x), int(corr_y), k]
            #     emptyImage[i, j, :] = pic[int(corr_x), int(corr_y), :]
            #     continue
            point1 = (math.floor(corr_x), math.floor(corr_y))  # 左上角的点
            point2 = (point1[0],  point1[1] + 1)
            point3 = (point1[0] + 1, point1[1])
            point4 = (point1[0] + 1, point1[1] + 1)

            fr1 = (point2[1]-corr_y)*pic[point1[0], point1[1], :] + (corr_y-point1[1])*pic[point2[0], point2[1], :]
            fr2 = (point2[1]-corr_y)*pic[point3[0], point3[1], :] + (corr_y-point1[1])*pic[point4[0], point4[1], :]
            emptyImage[i, j, :] = (point3[0]-corr_x)*fr1 + (corr_x-point1[0])*fr2

    return emptyImage


def nearest_resize(pic, scale):
    """
    nearest resize
    :param pic:  img
    :param scale:  scale >0
    :return:  resized img
    """

    h,w,c = pic.shape  # height,width,channel
    th, tw = int(h * scale), int(w * scale)
    # avoid the out of bounds from the original img
    pic = np.pad(pic, ((0, 1), (0, 1), (0, 0)), 'reflect')

    emptyImage = np.zeros((th, tw,c ), np.uint8)
    h_scale = h/th
    w_scale = w/tw
    for i in range(th):
        for j in range(tw):
            # 首先找到在原图中对应的点的(X, Y)坐标
            #first, find the location from the original img
            corr_x = (i + 0.5)*h_scale - 0.5
            corr_y = (j + 0.5)*w_scale - 0.5
            emptyImage[i, j, :] = pic[int(corr_x), int(corr_y), :]

    return emptyImage

def add_suffix(img_file,suffix):
    """
    add suffix for a given file name, and not change the file type
    :param img_file: img file, e.g. "xxx.jpg"
    :param suffix:  "——abcde"
    :return: "xxx——abcde.jpg"
    """
    name = os.path.splitext(img_file)[0]
    type_ = os.path.splitext(img_file)[1]
    file_name = name + suffix + type_
    return file_name


if __name__ == '__main__':
    """

    """
    print("openCV_version:",cv.__version__)

    parser = argparse.ArgumentParser(description="set blue")
    parser.add_argument("--img_file",type=str,default="./dogsmall.jpg",help="path of the img file")
    args = parser.parse_args()

    img = cv.imread(args.img_file)

  
    #1. Image Resizing
    ## 1.a nearest
    ne_img4X = nearest_resize(img, scale=4)
    cv.imwrite(add_suffix(args.img_file,"_nearest4X"),ne_img4X)

    ## 1.b nearest
    bi_img4X = bi_linear_resize(img, scale=4)
    cv.imwrite(add_suffix(args.img_file,"_bilinear4X"),bi_img4X)

   

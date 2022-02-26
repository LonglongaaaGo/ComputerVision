import cv2 as cv
import argparse
import os
import numpy as np


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

def conv2D(img,kernel,padding,stride):
    """
    convlution 2D
    :param img: img with size [h,w,c]
    :param kernel:  with size [kh,kw,c] or [kh,kw]
    :param padding: padding number
    :param stride: stride
    :return:  filted img
    """
    h,w,c = img.shape  # height,width,channel

    if kernel.ndim==2:
        kernel = np.expand_dims(kernel, axis=-1)
    kh,kw,kc = kernel.shape #kernel h, kernel w, kernel channel,
    assert kc == 1 or kc == c

    oh = int((h-kh+2*padding)/stride)+1 #out height
    ow = int((w-kw+2*padding)/stride)+1 #out width

    #out
    emptyImage = np.zeros((oh, ow, c), np.float)
    img = np.pad(img,((padding, padding), (padding, padding), (0, 0)), 'constant')

    for i in range(oh):
        for j in range(ow):
            i_idx = i*stride
            j_idx = j*stride
            tmp_out = img[i_idx:i_idx+kh,j_idx:j_idx+kw,:]*kernel
            for cc in range(c):
                emptyImage[i,j,cc] = np.sum(tmp_out[:,:,cc])

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

def get_box_filter(height,width):
    """
    :param height: kernel height
    :param width: kernel width
    :return:  the box fileter
    """
    box_kernel = np.ones((height,width),np.float)
    box_kernel = box_kernel/box_kernel.size

    return box_kernel


if __name__ == '__main__':
    """
    """
    print("openCV_version:",cv.__version__)

    parser = argparse.ArgumentParser(description="set blue")
    parser.add_argument("--img_file",type=str,default="./5_gt.png",help="path of the img file")
    args = parser.parse_args()

    img = cv.imread(args.img_file)


    #Image Filtering
    ne_img_down = nearest_resize(img, scale=0.5)
    cv.imwrite(add_suffix(args.img_file, "_nearest_img_down"), ne_img_down)
    #2.b Box filter
    box_kernel = get_box_filter(3,3)
    #2.a Convolution filter
    conv_img = conv2D(img, box_kernel, padding=1, stride=1)
    #2.c Anti-aliasing Filter
    conv_down_img = nearest_resize(conv_img, scale=0.5)
    cv.imwrite(add_suffix(args.img_file, "_Anti-aliasing_img"), conv_down_img)

# -*- coding: utf-8 -*-

import cv2
import numpy as np


def create_mask(path, frame_number):

    im = cv2.imread(path+str(frame_number).zfill(5)+'.png',cv2.IMREAD_UNCHANGED)
    im_to_display = cv2.imread(path+str(frame_number).zfill(5)+'.png', cv2.IMREAD_UNCHANGED)

    im_to_display_b = im_to_display.copy()

    im2 = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # lower_red = np.array([0, 0, 0])
    # upper_red = np.array([255, 255, 100])
    # mask = cv2.inRange(im2, lower_red, upper_red)

    # if frame_number==4:
    #     cv2.imshow('bgr', im2)
    #     cv2.imshow('gray', im_gray)
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(0)

    lower_red = np.array([200])
    upper_red = np.array([255])
    mask = cv2.inRange(im_gray, lower_red, upper_red)
    
    cv2.imwrite(path+'mask/'+str(frame_number).zfill(5)+'.png', mask)

    return mask

def create_consistency_img(out_path, consistency_path, frame_number):
    print('Out Path: ', out_path+'%s.png' % (str(frame_number).zfill(5)))
    print('Mask path: ', consistency_path+'mask/%s.png' % (str(frame_number+1).zfill(5)))
    # print('Consistency Path: ', consistency_path+'%s.png')


    out = cv2.imread(out_path+'%s.png' % (str(frame_number).zfill(5)))
    mask = cv2.imread(consistency_path+'mask/%s.png' % (str(frame_number+1).zfill(5)))

    consistency = out.copy()
    x,y = out.shape[:2]
    mask =  cv2.resize(mask, dsize=(y,x), interpolation=cv2.INTER_CUBIC)
    # print('mask: ', mask)

    consistency[mask.round()!=255] = 0
    # if frame_number==2:
    #     cv2.imshow('test', mask)
    #     cv2.imshow('cons', consistency)
    #     cv2.imshow('out', out)
    #     cv2.waitKey(0)

    cv2.imwrite(consistency_path+'%s.png' % (str(frame_number+1).zfill(5)), consistency)


def main():
    
    for i in range(4, 147):
        create_mask(0,i)
    
    
    
if __name__ == "__main__":
    main()
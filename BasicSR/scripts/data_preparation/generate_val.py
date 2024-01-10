import resize_right
import cv2
import os

GT_PATH = 'datasets/DF2K/val_gt'
LQ_PATH = 'datasets/DF2K/val_lq2'

images = os.listdir(GT_PATH)
for image in images:
    img = cv2.imread(os.path.join(GT_PATH, image))
    lq = resize_right.resize(img, scale_factors=0.5)
    cv2.imwrite(os.path.join(LQ_PATH, image), lq)

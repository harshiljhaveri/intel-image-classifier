
import cv2 as cv
import glob

imagepath = 'intel-image-classification/seg_test/seg_test'
imgs_names = glob.glob(imagepath+'*/*.jpg')
print(imgs_names)
for imgname in imgs_names:
    print(imgname)
    img = cv.imread(imgname)
    print(hi)
    if img is None:
        print(imgname)
        print(hi)
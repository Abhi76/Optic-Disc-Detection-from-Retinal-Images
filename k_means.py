import cv2
import numpy as np
import glob
import time as time
import os
path = 'dataset\images\*.*'
no = 1
directory = "segment_kmeans"
if not os.path.exists(directory):
    os.makedirs(directory)
for file in glob.glob(path):
    start = time.time()
    print(file)
    im = cv2.imread(file)
    im = cv2.resize(im,(350,234))
    im1 = cv2.imread(file)
    im1 = cv2.resize(im1,(350,234))
    img = cv2.imread(file,0)
    img = cv2.resize(img,(350,234))
    _,g,_ = cv2.split(im)
    kernel = np.ones((25,25),np.uint8)
    blur = cv2.dilate(g, kernel, iterations = 2)
    blur = cv2.GaussianBlur(blur,(55,55),20)
    #blur = cv2.medianBlur(blur,35)
    z = blur.reshape(-1,1)
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + 
    cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    K = 17

 
    ret,label,center=cv2.kmeans(z,K,None,criteria,100,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(g.shape)
    max = np.max(res2)
    m = res2 < max
    im[m] = 0
    fname = 'segment_kmeans/'+str(no)+'.jpg'
    cv2.imshow('original',im1)
    cv2.imwrite(fname,im)
    no=no+1
    cv2.imshow('blur',blur)
    cv2.imshow('segment',im)
    cv2.imshow('kmeans',res2)
    cv2.waitKey(1)
print('Running time: ')
rtime = time.time() - start
print(rtime)
cv2.waitKey(1)
cv2.destroyAllWindows()
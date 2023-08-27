import numpy as np
import skfuzzy as fuzz
import cv2
import re
import glob
import time as time
import os
numbers = re.compile(r'(\d+)')
path = 'dataset\images\*.*'
directory = "segment_fuzzyc"
if not os.path.exists(directory):
    os.makedirs(directory)
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts 

def change_color_fuzzycmeans(cluster_membership, clusters):
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

no=0
for file in sorted(glob.glob(path), key=numericalSort):
    start = time.time()
   
    print(file)
    image=cv2.imread(file)
    rgb_img=cv2.resize(image, (350,234))
    image=cv2.resize(image, (350,234))
    kernel = np.ones((21,21),np.uint8)
    b,g,r=cv2.split(image)
    rgb_img=g
    rgb_img = cv2.dilate(rgb_img, kernel, iterations=2)
    rgb_img = cv2.GaussianBlur(rgb_img,(45,45),14) 
    rgb_img = rgb_img.reshape((rgb_img.shape[0] * rgb_img.shape[1],1))
    img = np.reshape(rgb_img, (234,350)).astype(np.uint8)
    shape = np.shape(img)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(rgb_img.T, 17, 2, error=0.05, maxiter=100, init=None)
    new_img = change_color_fuzzycmeans(u,cntr)
    fuzzy_img = np.reshape(new_img,shape).astype(np.uint8)
    cv2.imshow('original',image)
    cv2.imshow('g',g)
    m = np.max(fuzzy_img)
    mask = fuzzy_img<m
    image[mask]=0 
    cv2.imshow('fcm',fuzzy_img)
    cv2.waitKey(1)
    cv2.imshow('segment',image)
    fname = 'segment_fuzzyc/seg'+str(no)+'.jpg'
    no=no+1
    cv2.imwrite(fname,image)
    cv2.waitKey(1)
print('Running time: ')
rtime = time.time() - start
print(rtime)
cv2.destroyAllWindows()
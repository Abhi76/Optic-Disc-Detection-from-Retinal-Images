import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd 
import glob
import re
import os

directory = "segmentdb"
if not os.path.exists(directory):
    os.makedirs(directory)

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
path = 'dataset/images/*.*'
no=0

# df = pd.read_csv("csv/size.csv", index_col =0)
for file in sorted(glob.glob(path),key =numericalSort):
    img=cv2.imread(file)
    img1=cv2.imread(file)
    img2 = cv2.imread(file)
    n = 0
    while(n<4):
        img = cv2.pyrDown(img)
        img1 = cv2.pyrDown(img1)
        if n<3:
            img2 = cv2.pyrDown(img2)
        n = n+1
    cv2.imshow('Original',img)
    kernel = np.ones((3,3),np.uint8)
    z=np.float32(img.reshape(-1,3))
    db = DBSCAN(eps=1.1, min_samples=7, metric = 'euclidean',algorithm ='auto').fit(z[:,:2])
    i = np.uint8(db.labels_.reshape(img.shape[:2]))
    m = np.max(i)
    mask = i < m
    img1[mask] = 0
    img1 = cv2.erode(img1, kernel, iterations = 1)
    img1 = cv2.pyrUp(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img1,127,255,0)
    # cv2.imshow("thres",thresh)
    cv2.waitKey(1)
    M = cv2.moments(thresh)
    if M["m00"] !=0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if cX < img1.shape[1]/2:
            cX = cX-9
        else:
            cX = cX+9
        cv2.circle(img2, (cX, cY), 25, (0,255,0), 2)
    else:
        cX = 0
        cY = 0
    
    fname = 'segmentdb/img'+str(no)+'.jpg'
    cv2.imshow('optic_disc',img2)
    cv2.imwrite(fname,img2)
    no=no+1
    # cv2.imshow('i',img1)
    diameter = 0
    diameter = np.sqrt(4 * M['m00'] / np.pi)
    df1 = pd.DataFrame({"Diameter": [diameter*8],"X_Center":[cX*8],"Y_Center":[cY*8]}) 
    # df = df.append(df1, ignore_index = True)
    print("Center:", end=' ')
    print(cX*8, end=' ')
    print(cY*8)
    # print("Diameter:", end=' ')
    # print(diameter)
cv2.waitKey(1)
export_csv = df1.to_csv (r'size.csv')
cv2.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import time as time
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import os
directory = "segment_aglomerative"
if not os.path.exists(directory):
    os.makedirs(directory)
num=0
no=1
sum=0
for file in glob.glob('dataset/images/*.*'):
    orig_img = cv2.imread(file)
    rorig_img=cv2.resize(orig_img,(350,233))
    cv2.imshow('original image',rorig_img)
    # gray_img=cv2.cvtColor(orig_img,cv2.COLOR_BGR2GRAY)
    _,gray_img,_ = cv2.split(orig_img)
    rescaled_img=cv2.resize(gray_img,(350,233))
    rescaled_img=cv2.GaussianBlur(rescaled_img,(23,23),38)
    kernel = np.ones((39,39),np.uint8)#33,33
    dilation_img=cv2.dilate(rescaled_img,kernel,iterations = 3)
    dilation_img=cv2.erode(dilation_img,kernel,iterations = 2)
    # dilation_img=cv2.GaussianBlur(dilation_img,(23,23),1)
    cv2.imshow("dila",dilation_img)
    X = np.reshape(dilation_img, (-1, 1))
    # Define the structure A of the data. Pixels connected to their neighbors.
    connectivity = grid_to_graph(*dilation_img.shape)
    # Compute clustering
    print("Compute structured hierarchical clustering...")
    st = time.time() 
    n_clusters = 15 # number of regions
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
    ward.fit(X)
    print(ward.labels_, dilation_img.shape)
    label = np.reshape(ward.labels_, dilation_img.shape)
    t=time.time()-st
    30
    print("Elapsed time: ", t) 
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)
    sum=sum+t
    # Plot the results on an image
    plt.figure(figsize=(5, 5))
    plt.imshow(dilation_img, cmap=plt.cm.gray)
    for l in range(n_clusters):
        plt.contour(label == l,colors=[plt.cm.nipy_spectral(l / float(n_clusters)), ])
    # plt.show()
    m=np.max(dilation_img)
    mask=dilation_img<m
    rorig_img[mask]=0
    #plt.imshow(rorig_img, cmap= plt.cm.gray)
    cv2.imshow('asdsa',rorig_img)
    #cv2.imwrite('C:/Users/Hirak/Desktop/01_h_mask.jpg',rorig_img)
    if num==1:
        cv2.waitKey(15000)
    num=num-1
    fname='segment_aglomerative/'+str(no)+'.jpg'
    cv2.imwrite(fname,rorig_img)
    no=no+1
    cv2.waitKey(1)
print('Total time') 
print(sum)
cv2.waitKey(1)
cv2.destroyAllWindows()
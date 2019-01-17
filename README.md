# Content Intro
This repository contains some different ways of methods personal uesd to realize the radar HRRP target classfication. Including CNN 、2channel CNN、DAE、SDAE based on tensorflow frame work and python. 
# Guide
folder and its content

CNN - Convolutional neural network

DCNN - Double channel convolutional neural network

DAE - Denoise autoencoder

SDAE - Stacked denoise autoencoder

# About data
Sample data are given:

Train_hrrp.mat —— struct:{name=aa} train data, created by Matlab, will be loaded as a structure by python, the data format is [row = sample, col = data], data = [onehot labels(x3) , HRRP data(x256)] 


Test_hrrp.mat —— struct:{name=bb} test data, created by Matlab, will be loaded as a structure by python,the data format is [row = sample, col = data], data = [onehot labels(x3) , HRRP data(x256)] 

# Contact me
feel free to contact me

My name is Alex （or Zikun Xu)

My mail :zikun19961215@foxmail.com    QQ:854587355

# Can I get your stars
if you find my share useful and helpful, please star me, thank you.



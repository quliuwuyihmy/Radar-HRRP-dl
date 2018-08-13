# Content Intro
This repository contains some different ways of methods personal uesd to realize the radar HRRP target classfication. Including CNN 、2channel CNN、DAE、SDAE based on tensorflow frame work and python. At the same time, some preprosessing methods of HRRP data are  included, such as PCA、 LDA， and the tool kit I used to do this prepresessing is shared in Matlab.
# Guide
folder and its content

CNN - Convolutional neural network

DCNN - Double channel convolutional neural network

DAE - Denoise autoencoder

SDAE - Stacked denoise autoencoder

# About data
I have to give my apologize for not offering the whole version of HRRP data that is needed for the project, for the data I am using is actually not allowed to open publicly.

the limited dataset offerred is HRRP data of three air-plane （no more detail , apologize)


Train_hrrp.mat —— struct:{name=aa} train data, created by Matlab, will be loaded as a structure by python, the data format is [row = sample, col = data], data = [onehot labels(x3) , HRRP data(x256)] 


Test_hrrp.mat —— struct:{name=bb} test data, created by Matlab, will be loaded as a structure by python,the data format is [row = sample, col = data], data = [onehot labels(x3) , HRRP data(x256)] 

# Contact me
Contact me if you have new idea on my code or simply you need my help.

My name is Alex 

My mail :854587355@qq.com

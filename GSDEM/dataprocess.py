# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:51:59 2020
@author: ZhangComputer
"""
from osgeo import gdal
import numpy as np
import os 
from glob import glob
from math import ceil
import time
import h5py
from osgeo import gdal, gdal_array
# import matplotlib.pyplot as plt
import cv2
from sklearn import metrics
import matplotlib.pylab as plt#画图模块



def compare(imgconpare,dst1):
    #dst1=dst[:,:,1]
    dstflat=dst1.flatten()
    imgconpareflat=imgconpare.flatten()
    MSE = metrics.mean_squared_error(dstflat, imgconpareflat)
    RMSE = metrics.mean_squared_error(dstflat, imgconpareflat)**0.5
    MAE = metrics.mean_absolute_error(dstflat, imgconpareflat)
    R2=metrics.r2_score(dstflat, imgconpareflat)
    return MSE,RMSE,MAE,R2
def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    im_width = ds.RasterXSize#栅格矩阵的列数
    im_height = ds.RasterYSize#栅格矩阵的行数
    im_data = ds.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    ds = None
    return im_width,im_height,geotrans,im_data

def geoToPixel(record,GeoTransform):
    uper_left_x = float(GeoTransform[0])
    uper_left_y = float(GeoTransform[3])
    pixel_width = float(GeoTransform[1])
    pixel_height = float(GeoTransform[5])
    (mx,my) =(record[0],record[3])
    px = int((mx - uper_left_x) / pixel_width) #x pixel
    py = int((my - uper_left_y) / pixel_height) #y pixel
    return (px,py)

gdal.AllRegister()
path = 'your path'
os.chdir(path)
raster_list = sorted(glob('*.tif'))
print(raster_list)
shp_path = 'your path'
srcImage2 = gdal.Open(shp_path)
srcArray2=np.array(srcImage2.GetRasterBand(1).ReadAsArray())
geoTrans2 = srcImage2.GetGeoTransform()
# fx = 0.00027777778/0.00083333333
# fy = 0.00027777778/0.00083333333

size3=32
size4=480
size5=160

size=737
pile32=np.zeros(size*25*size3*size3,dtype=np.float32).reshape(size*25,1,size3,size3)
pile160=np.zeros(size*25*size4*size4,dtype=np.float32).reshape(size*25,1,size4,size4)
pile=np.zeros(size*25*size5*size5,dtype=np.float32).reshape(size*25,1,size5,size5)

fx = 0.00027777778/0.00083333333
fy = 0.00027777778/0.00083333333

#737  0.5
#928  0.2
#1019  没有
a=0
k=0
m=0
count=0
loca960=0
loca240=0
num=0
for infile in raster_list:
    im_width,im_height,geotrans,im_data = GetExtent(infile)
    if im_data.shape[1]==3601 and im_data.min()!=-32768.0:
        # img = np.dstack([im_data] * 3)
        # enlarge_CUBIC = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        # enlarge_CUBIC = enlarge_CUBIC[:, :, 1]
        # im_input = enlarge_CUBIC
        im_input=im_data[:3600,:3600]

        img = np.dstack([im_data] * 3)
        enlarge_CUBIC = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        enlarge_CUBIC = enlarge_CUBIC[:, :, 1]
        im_input2 = enlarge_CUBIC[:1200, :1200]


        uper_left_x = float(geotrans[0])
        uper_left_y = float(geotrans[3])
        pixel_width = float(geotrans[1])
        pixel_height = float(geotrans[5])
        low_right_x=uper_left_x+pixel_width*1200
        low_right_y=uper_left_y+pixel_height*1200
        print(infile,num)
        num=num+1

        # x, y = geoToPixel(geotrans, geoTrans2)
        # print(infile,x,y)
        # if x > 0 and y > 0:
        if 105 <uper_left_x < 160 and 105 <low_right_x < 160 and -46 <uper_left_y < -5 and -46 <low_right_y < -5:
            x, y = geoToPixel(geotrans, geoTrans2)
            clip = srcArray2[y:y + 240, x:x + 240]
            print(infile,x,y)

            img3 = np.dstack([clip] * 3)
            enlarge_CUBIC2 = cv2.resize(img3, (0, 0), fx=15, fy=15, interpolation=cv2.INTER_NEAREST)
            enlarge_CUBICtest = enlarge_CUBIC2[:, :, 1]
            MSE, RMSE, MAE, R2 = compare(im_input, enlarge_CUBICtest)
            if MSE>0 and R2>=0.5:#11598
                print("count,total", count)
                count = count + 1

                for i in range(1, 6):
                    inx = i * size4
                    for j in range(1, 6):
                        iny = j * size4
                        pile160[a, ...] = im_input[iny:iny + size4, inx:inx + size4]
                        a = a + 1
                for i in range(1, 6):
                    inx = i * size3
                    for j in range(1, 6):
                        iny = j * size3
                        pile32[k, ...] = clip[iny:iny + size3, inx:inx + size3]
                        k = k + 1

                for i in range(1, 6):
                    inx = i * size5
                    for j in range(1, 6):
                        iny = j * size5
                        pile[m, ...] = im_input2[iny:iny + size5, inx:inx + size5]
                        m = m + 1

            # if x > 0 and y > 0:
            # clip = srcArray2[y:y + 240, x:x + 240]
            # img3 = np.dstack([clip] * 3)
            # enlarge_CUBIC2 = cv2.resize(img3, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            # enlarge_CUBICtest = enlarge_CUBIC2[:, :, 1]
            # MSE, RMSE, MAE, R2 = compare(im_input, enlarge_CUBICtest)
            # if MSE>0 and R2>=0.3:#11598
print(count)
#2020
# pile160 = ((pile160 + 9612.0) / (4406.0 + 9612.0))
# print(pile160.shape)
# pile32 = ((pile32 + 9612.0) / (4406.0 + 9612.0))
# print(pile32.shape)
#2021
pile160 = ((pile160 + 9093.0) / (4404.0 + 9093.0))
print(pile160.shape)
pile32 = ((pile32 + 9093.0) / (4404.0 + 9093.0))
print(pile32.shape)
pile = ((pile + 9093.0) / (4404.0 + 9093.0))
print(pile.shape)

with h5py.File('your path', 'w') as f:
    f.create_dataset('160_160', data=pile)
    f.create_dataset('480_480',data=pile160)
    f.create_dataset('32_32',data=pile32)






        # img3 = np.dstack([clip] * 3)
        # enlarge_CUBIC2 = cv2.resize(img3, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        # enlarge_CUBICtest = enlarge_CUBIC2[:, :, 1]
        # MSE, RMSE, MAE, R2 = compare(im_input, enlarge_CUBICtest)
        # if MSE>0 and R2>=0.5:#11598
            # im_inputfalt = im_input.flatten()
            # im_inputfalt=im_inputfalt.reshape(960*960, 1)
            # NASA_im_input[loca960:loca960 + 960*960, :]=im_inputfalt
            #
            # enlarge_CUBICtestfalt = enlarge_CUBICtest.flatten()
            # enlarge_CUBICtestfalt = enlarge_CUBICtestfalt.reshape(960 * 960, 1)
            # NASA_CUBICtest[loca960:loca960 + 960 * 960, :] = enlarge_CUBICtestfalt
            #
            # clipfalt = clip.flatten()
            # clipfalt = clipfalt.reshape(240 * 240, 1)
            # NASA_clip[loca240:loca240 + 240 * 240, :] = clipfalt

            # loca960 = loca960 + 960 * 960
            # loca240 = loca240 + 240 * 240



            # for i in range(1, 5):
            #     inx = i * size4
            #     for j in range(1, 5):
            #         iny = j * size4
            #         pile160[a, ...] = im_input[iny:iny + size4, inx:inx + size4]
            #         a = a + 1
            # for i in range(1, 5):
            #     inx = i * size3
            #     for j in range(1, 5):
            #         iny = j * size3
            #         pile32[k, ...] = clip[iny:iny + size3, inx:inx + size3]
            #         k = k + 1











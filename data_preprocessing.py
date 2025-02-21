from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import typing
import pandas as pd
import numpy as np

import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import cv2


def create_split(data: pd.DataFrame) ->typing.Tuple[typing.List[str]]:
    data_shuffle = shuffle(data,random_state=42)  # set random seed for reproducibility

    # Hard-code oil instances to solve class imbalance
    oil_instances = data_shuffle['filename'][data_shuffle['fuel_type'].values == 'Fossil Oil'].values
    data_shuffle = data_shuffle[np.invert(np.isin(data_shuffle['filename'].values,oil_instances))] #remove oil instances
    N = data_shuffle.shape[0]

    train, val_test = train_test_split(data_shuffle['filename'].values, test_size=0.2, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42) #0.11 since it should contribute 10% of the whole dataset
    train = np.append(train,oil_instances[0])
    val = np.append(val,oil_instances[1])
    test = np.append(test,oil_instances[2])
    return train, val, test

def crop_image_and_segmentation(filepath : str, segmentation_path: str, size : int = 120, channels: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) -> tuple:  
    """Crops an image and a segmenation file (both in tif format) to a square of the specified size in the specified channels. 
    This code is very largely based on that by Hanna et al. (2023) from their paper ``Physics-Guided Multitask Learning for Estimating
    Power Generation and CO2 Emissions From Satellite Imagery''. Even if only a cropped image is desired, a 
    corresponding segmentation filepath should be given. This is because the decision on whether to use cubic interpolation to resolve
    any spatial resolution issues during cropping is based on the segmentation mask.

    Args:
        filepath (str): Path to image geoTIFF
        segmentation_path (str): Path to segmentation geoTIFF.
        channels (list, optional): List of indices of bands aka channels to use. Defaults to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13].

    Returns:
        tuple: Tuple of numpy.ndarrays containing cropped image and segmentation mask respectively in the specified channels.
    """
    imgfile = rio.open(filepath)
    imgdata = np.array([imgfile.read(i) for i in
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    # imgdata = imgdata[channels] #keep only wanted channels
    
    shape = imgdata.shape[1] #will be 300

    #force square:
    if imgdata.shape[2] != shape:
        newimgdata = np.empty((len(channels), shape, shape))
        newimgdata[:, :, :imgdata.shape[2]] = imgdata[:, :, :imgdata.shape[2]]
        newimgdata[:, :, imgdata.shape[2]:] = imgdata[:,:, imgdata.shape[2] - 1:]
        imgdata = newimgdata

    if segmentation_path is not None:
            
        seg_file = rio.open(segmentation_path)
        fptdata = seg_file.read() #there is only 1 band for the segmentation masks
        fptdata = fptdata.reshape(fptdata.shape[1:]) #drop the first dimension which is 1 anyway
        # print(f'fptdata has shape {fptdata.shape}, should be either (120,120) or (300,300).')

        if shape == 300: #300 x 300 images
            fptcropped = fptdata[int((fptdata.shape[0] - size) / 2):int((fptdata.shape[0] + size) / 2),
                                    int((fptdata.shape[1] - size) / 2):int((fptdata.shape[1] + size) / 2)] # crop segmentation data to right size
        
            if np.sum(fptcropped) == np.sum(fptdata): #if we effectively did not do any cropping on the segmentation mask 
                fptdata = fptcropped
                imgdata = imgdata[:, int((imgdata.shape[1] - size) / 2):int((imgdata.shape[1] + size) / 2),
                                    int((imgdata.shape[2] - size) / 2):int((imgdata.shape[2] + size) / 2)] # crop image to central 120x120 pixels 
            else: #if spatial resolution is different
                imgdata = cv2.resize(np.transpose(imgdata, (1, 2, 0)).astype('float32'), (size, size), #make sure the 300x300 image pixels are the first two dimensions and that bands is the last
                                        interpolation=cv2.INTER_CUBIC)
                imgdata = np.transpose(imgdata, (2, 0, 1)) #tranpose back to bands,pixel,pixel - now (13,size,size)
                fptdata = cv2.resize(fptdata.astype('float32'), (size, size), interpolation=cv2.INTER_CUBIC) #resize 

        return imgdata, fptdata
    
    else:
        if shape == 300: #300 x 300 images
           imgdata = imgdata[:, int((imgdata.shape[1] - 120) / 2):int((imgdata.shape[1] + 120) / 2),
                                int((imgdata.shape[2] - 120) / 2):int((imgdata.shape[2] + 120) / 2)] # crop image to central 120x120 pixels 
        
        return imgdata
        

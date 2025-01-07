from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import typing
import pandas as pd
import numpy as np

import json
import rasterio as rio
import torch
from rasterio.features import rasterize
from shapely.geometry import Polygon
# from torchvision import transforms
import cv2
import os

# from custom_augmentations import Flip, Mirror, Rotate
# from torch.utils.data import Dataset

def create_split(data: pd.DataFrame) ->typing.Tuple[typing.List[str]]:
    data_shuffle = shuffle(data,random_state=42)  # set random seed for reproducibility

    # Hard-code oil instances to solve class imbalance
    oil_instances = data_shuffle['filename'][data_shuffle['fuel_type'].values == 'Fossil Oil'].values
    data_shuffle = data_shuffle[np.invert(np.isin(data_shuffle['filename'].values,oil_instances))] #remove oil instances
    N = data_shuffle.shape[0]

    train_val, test = train_test_split(data_shuffle['filename'].values, test_size=0.1, random_state=42)
    train,val = train_test_split(train_val, test_size=0.11, random_state=42) #0.11 since it should contribute 10% of the whole dataset
    train = np.append(train,oil_instances[0])
    val = np.append(val,oil_instances[1])
    test = np.append(test,oil_instances[2])
    return train, val, test

def crop_image_and_segmentation(filepath : str, segmentation_path: str | None = None, channels: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]) -> tuple:  
    """_summary_

    Args:
        filepath (str): Path to image geoTIFF
        segmentation_path (str or None): Path to segmentation geoTIFF. Defaults to None in which case only the image is cropped.
        channels (list, optional): List of indices of bands aka channels to use. Defaults to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13].

    Returns:
        tuple: Tuple of numpy.ndarrays containing cropped image and segmentation mask respectively in the specified channels.
    """
    imgfile = rio.open(filepath)
    imgdata = np.array([imgfile.read(i) for i in
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]])
    # imgdata = imgdata[channels] #keep only wanted channels
    
    size = imgdata.shape[1] #will be 300

    #force square:
    if imgdata.shape[2] != size:
        newimgdata = np.empty((len(channels), size, size))
        newimgdata[:, :, :imgdata.shape[2]] = imgdata[:, :, :imgdata.shape[2]]
        newimgdata[:, :, imgdata.shape[2]:] = imgdata[:,:, imgdata.shape[2] - 1:]
        imgdata = newimgdata

    if segmentation_path is not None:
            
        seg_file = rio.open(segmentation_path)
        fptdata = seg_file.read() #there is only 1 band for the segmentation masks
        fptdata = fptdata.reshape(fptdata.shape[1:]) #drop the first dimension which is 1 anyway
        # print(f'fptdata has shape {fptdata.shape}, should be either (120,120) or (300,300).')

        if size == 300: #300 x 300 images
            fptcropped = fptdata[int((fptdata.shape[0] - 120) / 2):int((fptdata.shape[0] + 120) / 2),
                                    int((fptdata.shape[1] - 120) / 2):int((fptdata.shape[1] + 120) / 2)] # crop segmentation data to right size
        
            if np.sum(fptcropped) == np.sum(fptdata): #if we effectively did not do any cropping on the segmentation mask 
                fptdata = fptcropped
                imgdata = imgdata[:, int((imgdata.shape[1] - 120) / 2):int((imgdata.shape[1] + 120) / 2),
                                    int((imgdata.shape[2] - 120) / 2):int((imgdata.shape[2] + 120) / 2)] # crop image to central 120x120 pixels 
            else: #if the fptdata was not 120x120 originally, we resize as such
                imgdata = cv2.resize(np.transpose(imgdata, (1, 2, 0)).astype('float32'), (120, 120), #make sure the 300x300 image pixels are the first two dimensions and that bands is the last
                                        interpolation=cv2.INTER_CUBIC)
                imgdata = np.transpose(imgdata, (2, 0, 1)) #tranpose back to bands,pixel,pixel - now (13,120,120)
                fptdata = cv2.resize(fptdata, (120, 120), interpolation=cv2.INTER_CUBIC) #resize 

        return imgdata, fptdata
    
    else:
        if size == 300: #300 x 300 images
           imgdata = imgdata[:, int((imgdata.shape[1] - 120) / 2):int((imgdata.shape[1] + 120) / 2),
                                int((imgdata.shape[2] - 120) / 2):int((imgdata.shape[2] + 120) / 2)] # crop image to central 120x120 pixels 
        
        return imgdata
        


def whatthefuck(seg_path, data_path):
    default_transform = rio.transform.from_bounds(0, 0, 120, 120, width=120, height=120)
    seglabels = []
    segfile_lookup = {}

    for i, seglabelfile in enumerate(os.listdir(seg_path)):
        segdata = json.load(open(os.path.join(seg_path,
                                                seglabelfile), 'r'))
        seglabels.append(segdata)
        segfile_lookup[
            "-".join(segdata['data']['image'].split('-')[1:]).replace(
                '.png', '.tif')] = i

    seglabels_poly = []
    # read in image file names for positive images
    idx = 0
    for root, _, files in os.walk(data_path):
        for filename in files:
            if not filename.endswith('.tif'):
                continue
            if filename not in segfile_lookup.keys():
                continue
            img_file = os.path.join(root, filename)

            # extracting image size
            if "120x120" in root:
                size = 120
            elif "300x300" in root:
                size =300
            else: 
                print("Outlier size image")

            polygons = []
            for completions in seglabels[segfile_lookup[filename]]['completions']:
                for result in completions['result']:
                    polygons.append(
                        np.array(
                            result['value']['points'] + [result['value']['points'][0]]) * size / 100)
                    # factor necessary to scale edge coordinates
                    # appropriately
            # if 'positive' in root and polygons != []:
            #     seglabels_poly.append(polygons)
            #     idx += 1
            # elif 'negative' in root:
            #     seglabels_poly.append([])
            #     idx +=1


        # rasterize segmentation polygons
            fptdata = np.zeros(img_file.shape[1:], dtype=np.uint8)
            # polygons = seglabels_poly.copy()
            shapes = []

            if len(polygons) > 0:
                for pol in polygons:
                    try:
                        pol = Polygon(pol)
                        shapes.append(pol)
                    except ValueError:
                        continue
                fptdata = rasterize(((g, 1) for g in shapes),
                                    out_shape=fptdata.shape,
                                    all_touched=True)

            # convert raster to tiff 
            mask_name = f"data/labels/{filename}.tif"
            with rio.open(mask_name, 
                        'w',
                        driver='GTiff',
                        width=size,
                        height=size,
                        # dtype=load_file.dtype,
                        transform=default_transform,  # Adding wrong geotransform to avoid NotGeoreferencedWarning
                        count=13) as dst:
                dst.write(fptdata)  # writing

                # list_polygons = [pol.tolist() for pol in polygons]

        # if size == 300:
        #     fptcropped = fptdata[int((fptdata.shape[0] - 120) / 2):int((fptdata.shape[0] + 120) / 2),
        #                          int((fptdata.shape[1] - 120) / 2):int((fptdata.shape[1] + 120) / 2)]
        #     if np.sum(fptcropped) == np.sum(fptdata):
        #         fptdata = fptcropped
        #         imgdata = imgdata[:, int((imgdata.shape[1] - 120) / 2):int((imgdata.shape[1] + 120) / 2),
        #                           int((imgdata.shape[2] - 120) / 2):int((imgdata.shape[2] + 120) / 2)]
        #     else:
        #         imgdata = cv2.resize(np.transpose(imgdata, (1, 2, 0)).astype('float32'), (120, 120),
        #                              interpolation=cv2.INTER_CUBIC)
        #         imgdata = np.transpose(imgdata, (2, 0, 1))
        #         fptdata = cv2.resize(fptdata, (120, 120), interpolation=cv2.INTER_CUBIC)
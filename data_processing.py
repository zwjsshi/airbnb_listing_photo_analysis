# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:20:18 2020

@author: zewen
"""
import cv2 as cv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from PIL import Image, ImageStat
from sklearn.cluster import KMeans
import time
import mahotas as mt
import statsmodels.api
import string 

#import images 
path = 'C:\\Users\\zewen\\Desktop\\Computer Vision Marketing\\Project\\final\\images\\'
image_data = pd.read_csv(r'C:\\Users\\zewen\\Desktop\\Computer Vision Marketing\\Project\\final\\new_york_bed_data.csv')
#fill the 5 missing value from price column
#image_data['Price'].fillna((image_data['Price'].mean()), inplace=True)
#rearrange final table
list(image_data.columns.values) 
image_data = image_data[['Unnamed: 0',
 'Neighbourhood Cleansed',
 'classes of rating',
 'class of price',
 'predictors',
 'Review Scores Rating',
 'Price',
 'Bedrooms',
 'Beds',
 'Number of Reviews']]
image_data = image_data.rename(columns={'Unnamed: 0':'imageID'})

#dummy top 10 objects 
df = image_data['predictors']
#remove punctuation for predictors
#since predictors are string rather than list with tuples
punc = set(string.punctuation)
convert = []
for i in image_data['predictors']:
    i = ''.join(ch for ch in i if ch not in punc)
    i = i.split()
    convert.append(i)
#define a function to get the string 
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
eachPredictor = []
allPredictor = []
for k in convert:
    for j in k:
        if hasNumbers(j) is False:
            eachPredictor.append(j)
    allPredictor.append(eachPredictor)
    eachPredictor = []
#concentrat list and find unique values
predictorConcat = []
for predictor in allPredictor:
    predictorConcat = predictorConcat + predictor 
predictorConcat_unique = pd.Series(predictorConcat).unique()
#count frequencies for predictors
predictor_freq = pd.DataFrame(0, index=np.arange(len(allPredictor)), columns = predictorConcat_unique)
#count the frequency 
for i in range(0,len(allPredictor)):
    for j in range(0,len(predictorConcat_unique)):
        if(str(allPredictor[i]).find(predictorConcat_unique[j])!=-1):
            predictor_freq.set_value(i, predictorConcat_unique[j],1)      
#top 10 frequencies 
predictor_sum = predictor_freq.sum()
top10 = predictor_sum.sort_values(ascending=False).head(10)
top10predictor = top10.index
top10
# =============================================================================
# fourposter     1724
# studiocouch    1723
# quilt          1554
# mosquitonet    1400
# wardrobe       1355
# slidingdoor    1309
# windowshade    1280
# shoji           667
# hometheater     606
# crib            588
# =============================================================================
#########################recommendation test###################################
ny_bed_test = image_data
ny_bed_test['predictors'] = allPredictor
test_high_rate = ny_bed_test[ny_bed_test['classes of rating'] == 'High']
test_low_rate = ny_bed_test[ny_bed_test['classes of rating'] == 'Low']
predictor_freq_high = pd.DataFrame(0, index=np.arange(len(test_high_rate)), columns = predictorConcat_unique)
predictor_freq_low = pd.DataFrame(0, index=np.arange(len(test_low_rate)), columns = predictorConcat_unique)
for i in range(0,len(test_high_rate)):
    for j in range(0,len(predictorConcat_unique)):
        if(str(list(test_high_rate['predictors'])[i]).find(predictorConcat_unique[j])!=-1):
            predictor_freq_high.set_value(i, predictorConcat_unique[j],1)
for i in range(0,len(test_low_rate)):
    for j in range(0,len(predictorConcat_unique)):
        if(str(list(test_low_rate['predictors'])[i]).find(predictorConcat_unique[j])!=-1):
            predictor_freq_low.set_value(i, predictorConcat_unique[j],1)
#top 10 frequencies 
predictor_sum_high = predictor_freq_high.sum()
top20_high = predictor_sum_high.sort_values(ascending=False).head(20)
top20_high
predictor_sum_low = predictor_freq_low.sum()
top20_low = predictor_sum_low.sort_values(ascending=False).head(20)
top20_low
###############################################################################

#get the dummy for top 5 predictor 
top10_dummy = predictor_freq[['fourposter','studiocouch','quilt','mosquitonet','wardrobe',
                              'slidingdoor','windowshade','shoji','hometheater','crib']]

#set dataframes for storing images' features 
hog_df = pd.DataFrame(columns=['hog_0-20','hog_20-40','hog_40-60','hog_60-80'
                               ,'hog_80-100','hog_100-120','hog_120-140','hog_140-160'])
brightness_df = pd.DataFrame(columns=['brightness'])
corners_df = pd.DataFrame(columns=['corners'])
dom_color_df = pd.DataFrame(columns=['red','green','blue'])
edges_df = pd.DataFrame(columns=['edges'])
lines_df = pd.DataFrame(columns=['lines'])
texture_df = pd.DataFrame(columns=['dim1','dim2','dim3','dim4','dim5','dim6'
                                   ,'dim7','dim8','dim9','dim10','dim11','dim12','dim13'])

#defines function for feature extraction 
def getcorners(img):
    # convert image to gray scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    # detect corners with the goodFeaturesToTrack function. 
    corners = cv.goodFeaturesToTrack(gray, 1000, 0.01, 10) 
    corners = np.int0(corners) 
    number_of_corners = np.shape(np.squeeze(corners))[0]
    return number_of_corners
def getedges(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    number_of_edges = len(edges)
    return number_of_edges
def getlines(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250) 
    number_of_lines = len(lines)
    return number_of_lines
def getHog(img):
    fd = hog(img, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)
    fd = fd.reshape(-1,8)
    #print(fd.shape)
    return np.mean(fd,axis=0)
def getTexture(img):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(img)
    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean
def getBrightness(img_id):
    # opens image in current working directory, converts to greyscale, and pulls a float value for brightness
    img_src = Image.open(img_id).convert('L')
    stat = ImageStat.Stat(img_src)
    brightness = stat.mean[0]
    return brightness
def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return centroids[np.argmax(hist)]
def getDominantColor(img_id):
    image = cv.imread(img_id)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    scale_percent = 30 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    reshape = resized.reshape((resized.shape[0] * resized.shape[1], 3))
    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    return np.uint8(visualize)

#loop for set values of features into each dataframe
image_start_id = 1
image_end_id= len(image_data)
t0= time.clock()
for i in range(image_start_id,image_end_id+1):
    img_id = 'image_'+str(i)+'.jpg'
    #print(img_id)
    #print(path+img_id)
    img = cv.imread(path+img_id)
    #hog_df.append(getHog(img))
    hog_df.loc[len(hog_df)] = getHog(img)
    brightness_df.loc[len(brightness_df)] = getBrightness(path+img_id)
    corners_df.loc[len(corners_df)] = getcorners(img)
    edges_df.loc[len(edges_df)] = getedges(img)
    lines_df.loc[len(lines_df)] = getlines(img)
    texture_df.loc[len(texture_df)] = getTexture(img)
    dom_color_df.loc[len(dom_color_df)] = getDominantColor(path+img_id)
#check the loop time 
t= time.clock() - t0
print('time ',t)

#test whether rows of dataframes are same  
dom_color_df.shape[0] == hog_df.shape[0] == brightness_df.shape[0] \
    == corners_df.shape[0] == edges_df.shape[0] == lines_df.shape[0] \
        == texture_df.shape[0] == image_data.shape[0] == top10_dummy.shape[0] #True

#make all features in one dataframe 
image_features = pd.concat([dom_color_df,brightness_df,corners_df,edges_df,lines_df,hog_df,texture_df], axis=1)
image_features.head(5)
image_features.describe()
#check if there are null values 
image_features.isnull().sum().sum() #0

#make all features in one dataframe 
ny_bed = pd.concat([image_data,top10_dummy,image_features], axis=1).drop(['predictors'],axis=1)
#check if there are null values 
ny_bed.isnull().sum().sum() #5
ny_bed_final = ny_bed.dropna()

ny_bed_final.isnull().sum().sum() #0

#download ny_bed_final as csv
ny_bed_final.to_csv(r'C:\\Users\\zewen\\Desktop\\ny_bed_final.csv')

# =============================================================================
# count_nan = len(y) - image_features.count()
# count_nan
# =============================================================================
# =============================================================================
# y = image_data.Price[0:image_end_id,]
# y= np.array(y)
# X = np.array(image_features.values)
# X = np.float32(X)
# #image_features.corr()
# np.count_nonzero(np.isnan(y))
# #y[np.isnan(y)] = 140
# np.mean(y)
# regression = statsmodels.api.OLS(y,X)
# model = regression.fit()
# model.summary()
# =============================================================================
# SAUVC24_MCU
Firmware in Arduino Framework for AUV Marty's onboard MCU



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

pd.set_option('display.max_rows' , None)
pd.set_option('display.max_columns' , None)
#load data
df  = pd.read_csv("modified_iris.csv")
features = df.dropna()
features = features.loc[: , "SepalLengthCm":"PetalWidthCm"]
print(features.shape)

cor = features.corr()
print(cor)
print("Speal Length & Petal Length have the highest correlation")
print("Sepal Length & PetalWidth have the lowest correlation")
plt.figure(figsize=(13,13))
plt.imshow(features, cmap='RdYlBu')
plt.colorbar(label='Color Scale')
plt.title("Before Wrangling")
plt.xticks(range(len(features.columns)) , features.columns , rotation = 90)
plt.yticks(range(len(features.index)) , features.index )
plt.show()


for i in features.columns.tolist() :
    q1 = features[i].quantile(0.25)
    q3 = features[i].quantile(0.75)
    iqr = q3-q1
    lower = q1 - 1.5 * iqr
    higher = q3 + 1.5 * iqr
    features = features[(features[i] >= lower) & (features[i] <= higher)]

plt.figure(figsize=(13,13))
plt.imshow(features, cmap='RdYlBu')
plt.colorbar(label='Color Scale')
plt.title("After Wrangling")
plt.xticks(range(len(features.columns)) , features.columns , rotation = 90)
plt.yticks(range(len(features.index)) , features.index )
plt.show()

mean = []
median = []
mode = []
max = []
min = []
std = []
for i in features.columns.tolist() :
    mean.append(features[i].mean())
    median.append(features[i].median())
    mode.append(float(features[i].mode()))
    max.append(features[i].max())
    min.append(features[i].min())
    std.append(features[i].std())
d = pd.DataFrame(index = features.columns.tolist() , columns = ["Mean" , "Median" , "Mode" , "Max" , "Min" , "Std"] )
d["Mean"] = mean
d["Median"] = median
d["Mode"] = mode
d["Max"] = max
d["Min"] = min
d["Std"] = std

print(d)
labels = features.columns.tolist()
for i in range(features.shape[1]) :
    for j in range(features.shape[1]) :
        if j!=i and j > i :
            plt.figure(figsize=(5,5))
            x = np.array(features.iloc[: , i])
            y = np.array(features.iloc[: , j])
            plt.scatter(x,y)
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            plt.title(labels[i]+" vs "+labels[j])
features.hist(bins=5 , alpha = 0.25)

plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

def distance(i , j):
    return np.linalg.norm(i -j , ord = 2)
def nearest(i , arr):
    d = np.array([distance(i[:4],j) for j in arr])
    return np.argmin(d)+1
def update_centroids(data , k ):
    labels = range(1,k+1)
    ctr = []
    for i in labels :
        df = data[data["Label"] == i]
        print(df.shape)
        arr = np.array(df.loc[:,"SepalLengthCm":"PetalWidthCm"])
        mean = np.mean(arr, axis=0)
        # print(mean)
        ctr.append(mean)
    return ctr
def kmeans(k , feat , epsilon = 0.01):
    data = feat
    centroids = data.sample(n=k, random_state=1)
    # print(centroids)
    data["Label"] = [None]*data.shape[0]
    l = 1
    for i in centroids.index.tolist():
        features.loc[i , "Label"] = l
        l+=1
    centroids = np.array(centroids)
    # centroids = centroids[:4]
    # print(centroids)
    while(1):
        for index , row in data.iterrows():
            i = np.array(row)
            l = nearest(i , np.array(centroids))
            data.loc[index , "Label"] = l
        ctr = update_centroids(data , k )
        # print(ctr)
        # print(centroids)
        d = np.abs(ctr - centroids)
        if(np.mean(d) >epsilon ):
            centroids = ctr
        else :
            return data
            break




%% PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Functions
#standardizing data
def standardize(data):
    mean = np.mean(data , axis = 0 )
    std = np.std(data , axis = 0)
    std_data = (data - mean)/std
    return std_data

#Select k largest eigven values from the eigenvalues of a matrix
def keigenvec(covar , k):
    e = {}
    eigval , eigvec = np.linalg.eig(covar)
    eigval_i = np.argpartition(eigval , -k)[-k:]
   
    for i in eigval_i :
        e[eigval[i]] = eigvec[: , i]
       
    return e

#Principle Component Analysis Function
def dimensionality_reduction(data , k ) :
    if(k < data.shape[0]):
        data = standardize(data)
        covariance = np.cov(data , rowvar = False)
        k_eig = keigenvec(covariance , k)
        eigval = list(k_eig.keys())
        eigval = np.sort(eigval)
        E = np.array([k_eig[i] for i in eigval[::-1] ])
        E = E.transpose()
        return np.matmul(data , E)
    else :
        return None

## Main

df = pd.read_csv("iris.csv")
df.dropna(inplace = True)
#dropping species and ID columns
feat = df.loc[: , "SepalLengthCm":"PetalWidthCm"]
#Converting Datafram to numpy array
data = np.array(feat)


# Applying PCA
new_data = dimensionality_reduction(data , 2)
#Converting the new data in numpy format to dataframe
new_df = pd.DataFrame(new_data , columns = [0,1])
print(new_df)

x = new_data[:50 , 0 ]
y = new_data[:50 , 1]
plt.scatter(x,y,c="r")

x = new_data[50:100 , 0 ]
y = new_data[50:100 , 1]
plt.scatter(x,y,c="g")

x = new_data[100: , 0 ]
y = new_data[100: , 1]
plt.scatter(x,y,c="b")
plt.title("PLotting Datapoints after PCA")
plt.show()


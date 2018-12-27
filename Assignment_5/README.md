# Image Segmentation

Problem Statement
We intend to perform image segmentation. Image segmentation means that we can group similar pixels together and give these grouped pixels the same label.  
The grouping problem is a clustering problem. We want to study the use of K-means and Normalized -Cut methods on the Berkeley Segmentation Benchmark. 

### Dataset
1. Download the dataset and understand the format (5 Points)
a. We will use Berkeley Segmentation Benchmark
b. The data is available at the following link. http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz.  
c. The dataset has 500 images. The test set is 200 images only. We will report our results on the test set only.

Images vs ground-truth
![img_vs_grnd](https://github.com/youssef-ahmed/cs421-data-mining/blob/master/Assignment_5/screens/img_vs_grnd.png)

K-means (k=5)
![Kmeans](https://github.com/youssef-ahmed/cs421-data-mining/blob/master/Assignment_5/screens/Kmeans.PNG)


Normalized-cut for 5-NN graph (k=5)
![nc](https://github.com/youssef-ahmed/cs421-data-mining/blob/master/Assignment_5/screens/NC_5NN.PNG)

Normalized-cut VS K-means  

![nc_vs_km](https://github.com/youssef-ahmed/cs421-data-mining/blob/master/Assignment_5/screens/km_vs_nc_0.PNG)

![nc_vs_km](https://github.com/youssef-ahmed/cs421-data-mining/blob/master/Assignment_5/screens/km_vs_nc_1.PNG)


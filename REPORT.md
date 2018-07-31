
# **RoboND Perception Project: Object recognition and advanced pick & place**

[//]: # (Image References)

[img_intro_0]: imgs/gif_world2_motion.gif
[img_intro_1]: imgs/img_pointcloud_data.png
[img_intro_2]: imgs/img_pointcloud_data_labels.png
[img_pipeline]: imgs/img_perception_pipeline.png

[img_pipeline_filtering]: imgs/img_perception_pipeline_filtering.png
[img_cloud_input]: imgs/img_cloud_input.png
[img_sor_filtering]: imgs/img_sor_filtering.png
[img_downsampling]: imgs/img_downsampling.png
[img_cutting]: imgs/img_cutting.png
[img_ransac]: imgs/img_ransac.png


## **About the project**

This project consists in the implementation of the perception pipeline for a more advance **pick & place** task, which consists of a **PR2** robot picking objects from a table and placing them into the correct containers.

![PICK AND PLACE][img_intro_0]

We make use of the RGB-D camera in the **PR2** robot, which give us **Point Cloud** information ( very noisy in this case ) of the scene.

![POINT CLOUD DATA][img_intro_1]

Using this pointcloud data we apply the perception pipeline in order to recognize each object in the scene, as shown in the following picture.

![POINT CLOUD DATA LABELS][img_intro_2]

This project is the result of the **perception** lectures of the RoboND nanodegree, in which we learned the necessary steps to implement the perception pipeline, namely :

*   Filtering and segmentation techniques in pointclouds.
*   Clustering using the DBSCAN algorithm.
*   Feature engineering and classification using SVMs and Scikit-Learn.

The whole pipeline is shown in the following picture.

![PERCEPTION PIPELINE][img_pipeline]

In the following sections we will explain how we addressed each requirement of the project and its corresponding pipeline implementation, as well as explain the experiments we made and the results we got.

## **Filtering and RANSAC fitting**

This part of the pipeline is in charge of filtering the pointcloud, such that the result is a cloud containing only points that belong to the objects on top of the table.

![FILTERING PIPELINE][img_pipeline_filtering]

The steps implemented are the following :

*   Noise removal using **Statistical Outlier Removal**
*   Downsampling using **Voxel-grid downsampling**
*   Region cutting using **Passthrough Filtering**
*   Plane fitting using **RANSAC**

This filters are already implemented in the **pcl** library, and we used them using the **python-pcl** bindings.

### 1. _**Noise removal**_

The initial cloud has quite some noise that might get in the way. So, the first step implemented was to apply the Statistical Outlier Removal filter from **pcl** to remove the noise points in our original cloud.

![SOR filtering][img_sor_filtering]

### 2. _**Downsampling**_

![Downsample filtering][img_downsampling]

### 3. _**Cutting**_

![Passthrough filtering][img_cutting]

### 4. _**RANSAC**_

![RANSAC fitting][img_ransac]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  





# **RoboND Perception Project: Object recognition and advanced pick & place**

[//]: # (Image References)

[gif_intro]: imgs/gif_world2_motion.gif
[gif_filters_tuning]: imgs/gif_filters_tuning.gif

[img_intro_1]: imgs/img_pointcloud_data.png
[img_intro_2]: imgs/img_pointcloud_data_labels.png
[img_pipeline]: imgs/img_perception_pipeline.png

[img_pipeline_filtering]: imgs/img_perception_pipeline_filtering.png
[img_cloud_input]: imgs/img_cloud_input.png
[img_sor_filtering]: imgs/img_sor_filtering.png
[img_downsampling]: imgs/img_downsampling.png
[img_cutting]: imgs/img_cutting.png
[img_ransac]: imgs/img_ransac.png

[img_sor_intuition]: imgs/img_sor_intuition.png
[img_voxelgrid_intuition]: imgs/img_voxelgrid_intuition.png

## **About the project**

This project consists in the implementation of the perception pipeline for a more advance **pick & place** task, which consists of a **PR2** robot picking objects from a table and placing them into the correct containers.

![PICK AND PLACE][gif_intro]

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

This filters are already implemented in the **pcl** library, and we used them using the **python-pcl** bindings. The implementation of the filtering pipeline can be found in the file [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py).

### 1. _**Noise removal**_

The initial cloud has quite some noise that might get in the way. So, the first step was to apply the Statistical Outlier Removal filter from **pcl** to remove the noise points in our original cloud.

![SOR filtering][img_sor_filtering]

The way the filter works is by checking a number of neighboring points around each point of the pointcloud, and checking which of these points is at a given multiple of the standard deviation of the group analized. The ones that are closer than that measure are kept and the others are removed ( outliers ). More information can be found [**here**](http://pointclouds.org/documentation/tutorials/statistical_outlier.php).

![SOR intuition][img_sor_intuition]

The parameters we have to set for this filter are the **number of neighbors** to analyze of each point, and the **factor** of the standard deviation to use for the threshold.

The call to the pcl function is located in the **_denoise** method in the [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) file, and uses some tuned parameters for the filter options.

~~~python
    # Statistical Outlier Removal (SOR) params
    SOR_MEAN_K = 5
    SOR_THRESHOLD_SCALE = 0.001

    # ...

    """
    Applies statistical outlier removal
    :param cloud : the cloud to remove outliers from
    """
    def _denoise( self, cloud ) :
        _filter = cloud.make_statistical_outlier_filter()
        _filter.set_mean_k( PCloudFilterParams.SOR_MEAN_K )
        _filter.set_std_dev_mul_thresh( PCloudFilterParams.SOR_THRESHOLD_SCALE )

        return _filter.filter()
~~~

### 2. _**Downsampling**_

One way to save computation is working in a pointcloud with less datapoints ( kind of like working with a smaller resolution image ). This can be achieved by downsampling the pointcloud, which give us a less expensive cloud to make further computations.

![Downsample filtering][img_downsampling]

The filter that we used in this step of the pipeline is the **Voxel-grid Downsampling** filter from **pcl**, which allowed us to downsample the cloud by replacing the points inside a voxel by a single point. Basically, we are placing a grid of voxels of certain size ( **leaf size** ) around the pointcloud and replacing the points inside a voxel by a single representative point.

![VOXELGRID intuition][img_voxelgrid_intuition]

We used the **pcl**'s voxelgrid filter, which is used in the **_voxelGridDownsample** method in the [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) file, with some tuned leaf sizes.

```python
    # Voxel grid downsample params
    VOXEL_LEAF_SIZE = [ 0.01, 0.01, 0.01 ]

    # ...

    """
    Applies voxel grid downsampling to reduce the size of the cloud

    :param cloud : the cloud to apply downsampling to
    """
    def _voxelGridDownsample( self, cloud ) :
        _filter = cloud.make_voxel_grid_filter()
        _filter.set_leaf_size( PCloudFilterParams.VOXEL_LEAF_SIZE[0],
                               PCloudFilterParams.VOXEL_LEAF_SIZE[1],
                               PCloudFilterParams.VOXEL_LEAF_SIZE[2] )

        return _filter.filter()
```


### 3. _**Cutting**_

The next step in the pipeline is to isolate the objects and the tabletop in a single cloud, which involves removing all other points related to other elements.

To do this, the simplest technique is to use some information of the scene, which in this case is that the objects are in certain region in front of the robot. This allows us to basically **cut** that region and keep only those points for later usage.

The filter that implements this is the **Passthrough** filter, which cuts a whole **region** along a certain **axis** ( two more parameters to choose ).

![Passthrough filtering][img_cutting]

```python
    # Passthrough filter params - z
    PASSTHROUGH_AXIS_Z = 'z'
    PASSTHROUGH_LIMITS_Z = [ 0.608, 0.88 ]
    # Passthrough filter params - x
    PASSTHROUGH_AXIS_Y = 'y'
    PASSTHROUGH_LIMITS_Y = [ -0.456, 0.456 ]

    # ...

    """
    Applies 'cutting' ( passthrough filtering ) to the given cloud

    :param cloud : the cloud to apply cutting to
    """
    def _passThroughFiltering( self, cloud ) :
        _filter_z = cloud.make_passthrough_filter()
        _filter_z.set_filter_field_name( PCloudFilterParams.PASSTHROUGH_AXIS_Z )
        _filter_z.set_filter_limits( PCloudFilterParams.PASSTHROUGH_LIMITS_Z[0],
                                     PCloudFilterParams.PASSTHROUGH_LIMITS_Z[1] )

        _fcloud = _filter_z.filter()

        _filter_x = _fcloud.make_passthrough_filter()
        _filter_x.set_filter_field_name( PCloudFilterParams.PASSTHROUGH_AXIS_Y )
        _filter_x.set_filter_limits( PCloudFilterParams.PASSTHROUGH_LIMITS_Y[0],
                                     PCloudFilterParams.PASSTHROUGH_LIMITS_Y[1] )
        
        return _filter_x.filter()
```

### 4. _**RANSAC**_

![RANSAC fitting][img_ransac]

```python
    # RANSAC segmentation params
    RANSAC_THRESHOLD = 0.00545

    # ...

    """
    Applies RANSAC plane fitting to the given cloud

    :param cloud : the cloud to apply RANSAC to
    """
    def _ransacSegmentation( self, cloud ) :
        _segmenter = cloud.make_segmenter()
        _segmenter.set_model_type( pcl.SACMODEL_PLANE )
        _segmenter.set_method_type( pcl.SAC_RANSAC )
        _segmenter.set_distance_threshold( PCloudFilterParams.RANSAC_THRESHOLD )
        _tableIndices, _ = _segmenter.segment()

        # extract table and objects from inliers and outliers
        _tableCloud = cloud.extract( _tableIndices, negative = False )
        _objectsCloud = cloud.extract( _tableIndices, negative = True )

        return _tableCloud, _objectsCloud
```

### 5. _**Parameter Tuning**_

The previous filters mentioned had some parameters that had to be tuned in order to work correctly in the new setup. This is why we implemented a GUI tool to allow picking these parameters interactively.

![FILTERS TUNING TOOL][gif_filters_tuning]

## **Segmentation using Euclidean Clustering**



## **Feature extraction and object recognition**


### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  




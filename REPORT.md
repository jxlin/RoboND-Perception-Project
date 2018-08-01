
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
[img_pipeline_clustering]: imgs/img_perception_pipeline_clustering.png
[img_pipeline_features]: imgs/img_perception_pipeline_feature_extraction.png
[img_pipeline_svm]: imgs/img_perception_pipeline_classification.png

[img_sor_intuition]: imgs/img_sor_intuition.png
[img_voxelgrid_intuition]: imgs/img_voxelgrid_intuition.png
[img_passthrough_intuition]: imgs/img_passthrough_intuition.png
[img_ransac_intuition]: imgs/img_ransac_intuition.png
[img_cut_regions]: imgs/img_cut_regions.png

[img_clustering]: imgs/img_clustering.png
[img_classification]: imgs/img_cloud_clusters_labels.png

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

This filters are already implemented in the **pcl** library, and we used them with the **python-pcl** bindings. The implementation of the filtering pipeline can be found in the file [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py).

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

One way to save computation is working in a pointcloud with less datapoints ( kind of like working with a smaller resolution image ). This can be achieved by downsampling the pointcloud, which give us a less expensive cloud for the next stages of the pipeline.

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

![Passthrough filtering][img_cutting]

The filter that implements this is the **Passthrough** filter, which cuts a whole **region** along a certain **axis** ( two more parameters to choose ).

![Passthrough intuition][img_passthrough_intuition]

In our case we used two filters along two different axes, which allowed us to remove the small portions of the left and right table that appeared in the pointcloud.

![Passthrough regions][img_cut_regions]

We used 2 pcl's passthrough filter in this step ( **z** and **y** axes ) with limits tuned according to the scene. This implementation can be found in the **_passThroughFiltering** method in the [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) file.

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

The last step to get a cloud of only the objects in the table is to remove the table from the pointcloud. This can be achieved by extracting the points in the cloud that best fit a plane, which is the actual table ( again, using some information of the scene, as we know the table is a plane ).

The problem is that if we fit a plane to the whole cloud using something like least squares, we would end up fitting a plane that would go through the pointcloud in "**average**", which kind of like cuts the cloud in the middle.

To avoid this, we used the **RANdom SAmple Consensus** paradigm, which instead of fitting to the whole cloud, it fits to a smaller set that is in consensus with a given calculated plane. For more information, check [**this**](http://pointclouds.org/documentation/tutorials/random_sample_consensus.php) post.

![RANSAC intuition][img_ransac_intuition]

We made use of pcl's **RANSAC segmenter** with a parameter that represents the threshold for the consensus set, and got the following results ( extracted the table from the pointcloud, and then removed the table from the cloud ).

![RANSAC fitting][img_ransac]

The implementation of this step can be found in the **_ransacSegmentation** method, in the [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) file.

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

## **Segmentation using Euclidean Clustering**

The filtering portion of the pipeline gave us a filtered cloud that contains only points that belong to the objects on top of the table. The next step is to cluster this cloud into subclouds, each belonging to a single object.

![DBSCAN pipeline][img_pipeline_clustering]

Here we make use of the DBSCAN algorithm, also called Euclidean Clustering ( check [**here**](https://en.wikipedia.org/wiki/DBSCAN) and [**here**](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) for further info ), which allows us to cluster a group of datapoints by closeness ( distance ) and density, without specifying the number of cluster in the set.

![DBSCAN clustering][img_clustering]

We used the pcl's [**Euclidean Clustering**](http://pointclouds.org/documentation/tutorials/cluster_extraction.php) algorithm, which makes use of a **kdtree** for faster search, and has 3 parameters to tune.

*   Tolerance : distance a point can be to another to consider it neighbor in the cluster.
*   Min cluster size : minimum number of points that a cluster can have.
*   Max cluster size : maximum number of points that a cluster can have.

The implementation of this step of the pipeline can be found in the [**PCloudClusterMaker.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudClusterMaker.py) file, in the **_dbscan** method.

```python

    # DBSCAN params
    DBSCAN_CLUSTER_TOLERANCE = 0.025
    DBSCAN_MIN_CLUSTER_SIZE = 30
    DBSCAN_MAX_CLUSTER_SIZE = 2000

    # ...

    """
    Applies Euclidean Clustering to the given cloud and ...
    returns the cluster indices

    :param cloud : the cloud to apply the clustering to
    """
    def _dbscan( self, cloud ) :
        # transform the cloud to just an xyz cloud
        _xyzCloud = XYZRGB_to_XYZ( cloud )
        # create a kdtree for the cluster algorithm to use
        _kdtree = _xyzCloud.make_kdtree()
        # Create cluster extractor and configure its properties
        _dbscanExtractor = _xyzCloud.make_EuclideanClusterExtraction()
        _dbscanExtractor.set_SearchMethod( _kdtree )
        _dbscanExtractor.set_ClusterTolerance( PCloudClusterMakerParams.DBSCAN_CLUSTER_TOLERANCE )
        _dbscanExtractor.set_MinClusterSize( PCloudClusterMakerParams.DBSCAN_MIN_CLUSTER_SIZE )
        _dbscanExtractor.set_MaxClusterSize( PCloudClusterMakerParams.DBSCAN_MAX_CLUSTER_SIZE )

        return _dbscanExtractor.Extract()
```


## **Parameter Tuning**

The filters and clusterer mentioned earlier had some parameters that had to be tuned in order to work correctly in the new setup. This is why we implemented a GUI tool to allows picking these parameters interactively.

![FILTERS TUNING TOOL][gif_filters_tuning]

We reused some of the code made for the previous [**Kinematics**](https://github.com/wpumacay/RoboND-Kinematics-Project/blob/master/kuka_arm/scripts/utils/RIKpublisherUI.py) project; linked it to our actual perception pipeline, and made the GUI have control over the parameters of the pipeline, exposed as global static variables ( both sets of parameters can be found in the corresponding [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) and [**PCloudClusterMaker.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudClusterMaker.py) files )

```python

# PCloudFilter.py

class PCloudFilterParams :
    # Voxel grid downsample params
    VOXEL_LEAF_SIZE = [ 0.01, 0.01, 0.01 ]
    # Passthrough filter params - z
    PASSTHROUGH_AXIS_Z = 'z'
    PASSTHROUGH_LIMITS_Z = [ 0.608, 0.88 ]
    # Passthrough filter params - x
    PASSTHROUGH_AXIS_Y = 'y'
    PASSTHROUGH_LIMITS_Y = [ -0.456, 0.456 ]
    # RANSAC segmentation params
    RANSAC_THRESHOLD = 0.00545
    # Statistical Outlier Removal (SOR) params
    SOR_MEAN_K = 5
    SOR_THRESHOLD_SCALE = 0.001

# PCloudClusterMaker.py

class PCloudClusterMakerParams :
    # DBSCAN params
    DBSCAN_CLUSTER_TOLERANCE = 0.025
    DBSCAN_MIN_CLUSTER_SIZE = 30
    DBSCAN_MAX_CLUSTER_SIZE = 2000

```

The implementation of the tool can be found in the [**PParameterTunerUI.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PParameterTunerUI.py) file. 

Also, we checked the time cost that each step took ( just in case ), and got the following costs :

*   **Filtering** : _Denoising_ ~ 1300ms, _Downsampling_ ~ 36ms, _Passthrough_ ~ 0.5ms, _RANSAC_ ~ 1.8ms
*   **Clustering** : _DBSCAN_ ~ 8ms
*   **Classification** : _SVMpredict per object_ ~ 42ms

## **Feature extraction and object recognition**

The last step of the pipeline is to transform the clusters into feature vectors that can be fed into a classifier. In our case we use the pointclouds from the clustering process to make a single feature vector for each of the clusters, and then use an SVM to get the predictions.

### 1. _**Feature extraction**_

The features extracted consists of histograms for the colors and normals from the pointcloud of each cluster.

![FEATURES pipeline][img_pipeline_features]

We used the following approach :

*   **Colors histogram**: we used the **hsv** color space for the color histogram, and tuned the bining size according to some experiments with the SVM classifier. Also, we made sure to pass the ranges to compute the histogram ( ( 0, 255 ) in our case ).
*   **Normals histogram**: for the normals histogram we used the normals computed from a service already provided ( **'/feature_extractor/get_normals'** ), and also set the bining size according to some experiments with the SVM. The ranges for this case are ( -1, 1 ), as the vectors returned by the service are normalized.

Each histogram consists of 3 vectors ( 3 for **hsv** channels in the colors histograms, and 3 for **xyz** channels in the normals histograms ). We then just concatenate these into a single vector, which is the feature to be used for our classifier.

The implementation of the features extraction can be found in the [**PUtils**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PUtils.py) file ( lines )

We will later talk a bit more about the experimets run to tune the number of bins to use for each histogram, when we discuss some code implementation details.

### 2. _**Object recognition**_

With the features vector computed in the previous step we can finally apply our SVM classifier to get the predicted labels for each object.

![CLASSIFICATION pipeline][img_pipeline_svm]

We used Scikit-learn to create and train the SVM classifier, with the following parameters :

*   **Kernel**: linear
*   **C**: 1.0
*   **gamma**: 1.0
*   **dimensionality**: 1515 = _color_fvector_size_ + _normal_fvector_size_ = 255 * 3 + 250 * 3

The results we got for the previous clusters are shown in the following figure :

![CLASSIFICATION result][img_classification]

The classifier implementation can be found in the [**PClassifierSVM.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PClassifierSVM.py) file. We will talk more about the implementation later when we discuss some details about the code implementation of the project.

## **Pick and Place Setup**

The last step of the pipeline is to actually use the predictions of our classifier to trigger a pick-place operation. This can be achieved by sending the appropiate request to the pick-place service exposed in the project starting code.

Just to clarify, our pipeline is expected to work in 3 different environments, defined in the separate **.world** files :

*   `test1.world` : a scene with 3 objects on top of the table.
*   `test2.world` : the scame scene, with 5 objects on top of the table.
*   `test3.world` : the scame scene, with 8 objects on top of the table.

We can get the required objects to pick and where to place them from the respective `pick_list_*.yaml`, which is loaded into the ros parameter server.

This whole step is implemented in the [**PPickPlaceHandler.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PPickPlaceHandler.py) file, in which we do the following :

*   Load the required information from the parameter server ( **_initialize** method )
*   Handle a request from the pipeline to pick a list of classified objects ( **pickObjectsFromList** method )
*   Make single requests for the pick & place operation, by making the appropiate request messages ( **_pickObject** method ) and saving the **.yaml** dictionary version of the request, as required.

Below we show how me build the request message :

```python
    def _pickObject( self, pobject, callservice ) :
        # make service request
        _req = PickPlaceRequest()
        # scene number
        _req.test_scene_num.data = self.m_sceneNum
        # object name
        _req.object_name.data = pobject.label
        # arm name ( group )
        _req.arm_name.data = ( 'right' if pobject.group == 'green' else 'left' )
        # centroid
        _req.pick_pose.position.x = pobject.centroid[0]
        _req.pick_pose.position.y = pobject.centroid[1]
        _req.pick_pose.position.z = pobject.centroid[2]
        # drop position
        _req.place_pose.position.x = self.m_dropdict[ pobject.group ]['position'][0]
        _req.place_pose.position.y = self.m_dropdict[ pobject.group ]['position'][1]
        _req.place_pose.position.z = self.m_dropdict[ pobject.group ]['position'][2]
        # save yaml dict
        _yamlDict = make_yaml_dict( _req.test_scene_num,
                                    _req.arm_name,
                                    _req.object_name,
                                    _req.pick_pose,
                                    _req.place_pose )
        ## Send pick and place request #############

        if callservice :
            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service( 'pick_place_routine' )
            try:
                pick_place_routine = rospy.ServiceProxy( 'pick_place_routine', PickPlace )

                resp = pick_place_routine( _req.test_scene_num, 
                                        _req.object_name,
                                        _req.arm_name, 
                                        _req.pick_pose, 
                                        _req.place_pose )

                print ( "Response: ", resp.success )

            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        ############################################
**output1.yaml**
        return _yamlDict
```

And the `output*.yaml` files we got can be found in the following links :

*   [**output1.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output1.yaml)
*   [**output2.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output2.yaml)
*   [**output3.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output3.yaml)

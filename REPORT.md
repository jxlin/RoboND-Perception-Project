
# **RoboND Perception Project: Object recognition and advanced pick & place**

*   [**video 1 - just detection**](https://youtu.be/_-eAPI1gTGo)
*   [**video 2 - whole pick-place process**](https://youtu.be/Jqd7IYW6iMg)

---

## **Running the project**

### **Dependencies**

This project depends on the [**sensor_stick**](https://github.com/udacity/RoboND-Perception-Exercises/tree/master/Exercise-3/sensor_stick) package, which you can copy from [**this**](https://github.com/udacity/RoboND-Perception-Exercises/tree/master/Exercise-3/sensor_stick) repository to your catkin_ws.

### **Cloning the repo**
    # clone the repo
    cd YOUR_CATKIN_WS/src/
    git clone https://github.com/wpumacay/RoboND-Perception-Project.git
    cd .. 

    # Install dependencies
    rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
    
    # Adding the models to the path
    export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH

    # Build your workspace
    catkin_make

### Running the package

    # launch the .launch file for the project
    roslaunch pr2_robot pick_place_project.launch

    # Change to this location, as the path to the models data may not be loaded
    cd YOUR_CATKIN_WS/src/RoboND-Perception-Project/pr2_robot/scripts

    # Just in case, make sure the perception pipeline is executable
    chmod u+x perception_pipeline.py

    # Run the pipeline from there
    rosrun pr2_robot perception_pipeline.py

    # Or just run it like this
    ./perception_pipeline.py

### Configuration

Some configuration options are located in the [**pipeline.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/config/pipeline.yaml) file, and include :

*   **model** : one of the trained models in the **data/models** folder ( default -> model_klinear_c128_n50_sz2000_C10 )
*   **pickplace** : whether or not to enable pick-place requests to the service ( default -> False )
*   **save2yamls** : whether or not to save the result .dictionaries to a yaml file ( default -> True )

---

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

[img_histogram_hue_scaling]: imgs/img_histogram_scaling_hue.png
[img_histogram_nx_scaling]: imgs/img_histogram_scaling_nx.png

[img_results_sc1_klinear]: imgs/img_results_sc1_klinear.png
[img_results_sc2_klinear]: imgs/img_results_sc2_klinear.png
[img_results_sc3_klinear]: imgs/img_results_sc3_klinear.png

[img_results_sc1_kpoly]: imgs/img_results_sc1_kpoly.png
[img_results_sc2_kpoly]: imgs/img_results_sc2_kpoly.png
[img_results_sc3_kpoly]: imgs/img_results_sc3_kpoly.png

[img_cmatrix_linear]: imgs/img_confusion_matrix_chosen_linear_model.png
[img_cmatrix_poly]: imgs/img_confusion_matrix_chosen_poly_model.png

[gif_results_sc1]: imgs/gif_results_sc1.gif
[gif_results_sc2]: imgs/gif_results_sc2.gif
[gif_results_sc3]: imgs/gif_results_sc3.gif

[gif_pick_place_full]: imgs/gif_pick_place_full.gif
[gif_pickplace_operation_grasp_fail]: imgs/gif_pickplace_operation_grasp_fail.gif

[img_moveit_concepts]: imgs/img_moveit_concepts.png
[img_collisionmap_pipeline]: imgs/img_collision_avoidance_filtering.png

[img_collision_map_1]: imgs/img_collision_map_1.png

[img_statemachine_pick_place]: imgs/img_statemachine_pick_place.png

## **About the project**

This project consists in the implementation of the perception pipeline for a more advance **pick & place** task, which consists of a **PR2** robot picking objects from a table and placing them into the correct containers.

![PICK AND PLACE][gif_intro]

We make use of the RGB-D camera in the **PR2** robot, which give us **Point Cloud** information ( very noisy in this case ) of the scene.

![POINT CLOUD DATA][img_intro_1]

Using this pointcloud data we apply the perception pipeline in order to recognize each object in the scene, as shown in the following figures.

![POINT CLOUD DATA LABELS SC1][gif_results_sc1]

![POINT CLOUD DATA LABELS SC2][gif_results_sc2]

![POINT CLOUD DATA LABELS SC3][gif_results_sc3]

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

One way to save computation is working with a pointcloud with less datapoints ( kind of like working with a smaller resolution image ). This can be achieved by downsampling the pointcloud, which give us a less expensive cloud for the next stages of the pipeline.

![Downsample filtering][img_downsampling]

The filter that we used in this step of the pipeline is the **Voxel-grid Downsampling** filter from **pcl**, which allowed us to downsample the cloud by replacing the points inside a voxel by a single point. Basically, we are placing a grid of voxels of certain size ( **leaf size** ) around the pointcloud and replacing the points inside a voxel by a single representative point.

![VOXELGRID intuition][img_voxelgrid_intuition]

We used the **pcl**'s voxelgrid filter, which is used in the **_voxelGridDownsample** method in the [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) file, with some tuned leaf sizes.

```python
    # Voxel grid downsample params
    VOXEL_LEAF_SIZE = [ 0.0035, 0.0035, 0.0035 ]

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

The filter that implements this is the **Passthrough** filter, which cuts a whole **region** along a certain **axis** and between a range given by two values ( three more parameters to choose ).

![Passthrough intuition][img_passthrough_intuition]

In our case we used two filters along two different axes, which allowed us to remove the small portions of the left and right table that appeared in the pointcloud.

![Passthrough regions][img_cut_regions]

We used 2 pcl's passthrough filter in this step ( **z** and **y** axes ) with limits tuned according to the scene. This implementation can be found in the **_passThroughFiltering** method in the [**PCloudFilter.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PCloudFilter.py) file.

```python
    # Passthrough filter params - z
    PASSTHROUGH_AXIS_Z = 'z'
    PASSTHROUGH_LIMITS_Z = [ 0.608, 0.88 ]
    # Passthrough filter params - y
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
    VOXEL_LEAF_SIZE = [ 0.0035, 0.0035, 0.0035 ]
    # Passthrough filter params - z
    PASSTHROUGH_AXIS_Z = 'z'
    PASSTHROUGH_LIMITS_Z = [ 0.608, 0.88 ]
    # Passthrough filter params - y
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

Also, we checked the time cost that each step took ( just in case ), and got the ~~~following costs~~~ ( _**we tweaked a bit the parameters when we changed to a better linear model, that gave 8/8 in scene 3, and the pipeline might take a bit longer than the presented values**_ ) :

*   **Filtering** : _Denoising_ ~ 1300ms, _Downsampling_ ~ 36ms, _Passthrough_ ~ 0.5ms, _RANSAC_ ~ 1.8ms
*   **Clustering** : _DBSCAN_ ~ 8ms
*   **Classification** : _SVMpredict per object_ ~ 42ms

## **Feature extraction and object recognition**

The last step of the pipeline is to transform the clusters into feature vectors that can be fed into a classifier. In our case we use the pointclouds from the clustering process to make a single feature vector for each of the clusters, and then use an SVM to get the predictions.

### 1. _**Feature extraction**_

The features extracted consist of histograms for the colors and normals from the pointcloud of each cluster.

![FEATURES pipeline][img_pipeline_features]

We used the following approach :

*   **Colors histogram**: we used the **hsv** color space for the color histogram, and tuned the bining size according to some experiments with the SVM classifier. Also, we made sure to pass the ranges to compute the histogram ( ( 0, 255 ) in our case ).
*   **Normals histogram**: for the normals histogram we used the normals computed from a service already provided ( **'/feature_extractor/get_normals'** ), and also set the bining size according to some experiments with the SVM. The ranges for this case are ( -1, 1 ), as the vectors returned by the service are normalized.

Each histogram consists of 3 vectors ( 3 for **hsv** channels in the colors histograms, and 3 for **xyz** channels in the normals histograms ). We then just concatenate these into a single vector, which is the feature to be used for our classifier.

The implementation of the features extraction can be found in the [**PUtils**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PUtils.py) file ( lines 301-389 )

We will later talk a bit more about the experimets run to tune the number of bins to use for each histogram, when we discuss some code implementation details.

### 2. _**Object recognition**_

With the features vector computed in the previous step we can finally apply our SVM classifier to get the predicted labels for each object.

![CLASSIFICATION pipeline][img_pipeline_svm]

We used Scikit-learn to create and train the SVM classifier, with the following parameters :

*   **Kernel**: linear
*   **C**: 10.0
*   **dimensionality**: 534 = _color_fvector_size_ + _normal_fvector_size_ = 128 * 3 + 50 * 3

The results we got for the previous clusters are shown in the following figure :

![CLASSIFICATION result][img_results_sc3_klinear]

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

Below we show how we build the service request :

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
        ## start pick and place request ########################

        # rotate to account for the collision map


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
                return _yamlDict, resp.success

            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

        ########################################################

        return _yamlDict, None
```

And the `output*.yaml` files we got can be found in the following links :

*   [**output1.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output1.yaml)
*   [**output2.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output2.yaml)
*   [**output3.yaml**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output3.yaml)

## **SVM model selection**

There are some parameters that we had to choose to make our classifier, and they were the following :

*   Color space to use for color histograms
*   Bin sizes for the colors histograms
*   Bin sizes for the normals histograms
*   Amount of data to take from the training scene
*   SVM kernel and related parameters
*   SVM regularization parameter C

For the first one we found that the **HSV** colorspace worked better, so we kept using it. We tried first RGB and the accuracy for the RGB case, keeping the same configuration for both RGB and HSV cases, was lower by a big margin compared to the HSV case.

We tuned the other parameters by testing over the whole space given by these parameters. 

### _**Bin sizes and training batches**_

The way we started checking bin sizes was slow, as we had to choose the binsize and then take the training data for that size. If we wanted to test different bin sizes we had to start over and take the data again.

Because of this issue, we chose to first take a big batch of data with a **maximum** bin size, from which we could then tune it to a smaller value.

The big batch of data that we took can be found ( as a pickle .sav file ) in the **data/samples/** folder. The file we used for our experiments is the **training_set_2000.sav** file, which contains **2000** training samples for each category of the picklist ( 8 in our case ).

We then implemented the appropiate functionality to downsample our histograms to smaller binsizes, which can be found in the **hist2hist** method, in the [**PUtils.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PUtils.py) file.

```python
"""
Transforms one histogram to another with smaller bin size

: param hist : source histogram
: param nbins : target number of bins of the transformed histogram
"""
def hist2hist( hist, nbins ) :
    assert ( len( hist ) >= nbins )

    _rmin = np.min( hist )
    _rmax = np.max( hist )

    _newhist = np.zeros( nbins )
    _newedges = np.linspace( _rmin, _rmax, num = ( nbins + 1 ), endpoint = True )
    
    # compute bin sizes, new and old, for indexing
    _newbinsize = ( _rmax - _rmin ) / nbins
    _oldbinsize = ( _rmax - _rmin ) / len( hist )

    for i in range( nbins ) :
        _startIndx = int( math.floor( _newedges[i] / _oldbinsize ) )
        _stopIndx  = int( math.floor( _newedges[i + 1] / _oldbinsize ) - 1 )
        _newhist[i] = hist[ _startIndx : ( _stopIndx + 1 ) ].sum()

    return _newhist
```

Then, when doing the training experiments to tune the parameters of our model we chose different bin sizes taken from the base binsize, and then generated the appropiate batches of data for the experiments using the **_generateSessionBatch** method, in the [**PClassifierSVM.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PClassifierSVM.py) file.

```python
    """
    Generate new samples from the given samples by ...
    reshaping using different histograms sizes
    """
    def _generateSessionBatch( self, nbinsColors, nbinsNormals, dataX ) :
        # We start with a huge bag of data with ...
        # DATASET_COLOR_HIST_BINS, DATASET_NORMAL_HIST_BINS ...
        # as base size of the histograms. From there, we modify the ...
        # size according to the required sizes
        _batchX = []
        for _x in dataX :
            # extract color and normals histogram
            _x_chist = _x[0:(3 * DATASET_COLOR_HIST_BINS)]
            _x_nhist = _x[(3 * DATASET_COLOR_HIST_BINS):]
            _x_chist_channels = np.array_split( _x_chist, 3 )
            _x_nhist_channels = np.array_split( _x_nhist, 3 )
            # convert each histogram channel to the desired sizes
            _sx_chist_channels = [ hist2hist( _x_chist_channels[i], nbinsColors )
                                   for i in range( len( _x_chist_channels ) ) ]
            _sx_nhist_channels = [ hist2hist( _x_nhist_channels[i], nbinsNormals )
                                   for i in range( len( _x_nhist_channels ) ) ]
            _sx_chist = np.concatenate( _sx_chist_channels ).astype( np.float64 )
            _sx_nhist = np.concatenate( _sx_nhist_channels ).astype( np.float64 )
            _batchX.append( np.concatenate( [ _sx_chist, _sx_nhist ] ) )

        return np.array( _batchX )
```

For a testing script on how we scaled the bin sizes just check the [**test_features.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/tests/test_features.py) file. A sample is shown in the following figures:

![HISTOGRAM SCALING HUE][img_histogram_hue_scaling]

![HISTOGRAM SCALING NX][img_histogram_nx_scaling]

There we have the histograms for the **Hue** and **NormalX** parts of the feature vector, each with the base size of 255 and 250, respectively, and downsampled to 64 and 50.

### **_SVM model parameters_**

The other parameters we had to tune were the SVM parameters of the classifier, which consists of :

*   **Kernel** : _linear_, _rbf_ and _poly_. Depending of the kernel we may use the _gamma_ and _degree_ parameters ( if we use rbf or poly as kernels )
*   **Regularization** : Amount of regularization for our model

We explain in the next subsection how we tuned these parameters.

### **_Parameter tuning and experiments_**

Taking all the mentioned parameters into account we proceed to make experiments by running several training **sessions** and checking which combination yield better results. The full implementation of our experiments can be found in the [**PClassifierSVM.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PClassifierSVM.py) file.

We encapsulated all these parameters into a **session** object, which also includes other elements, like the data batch used, resulting accuracy, trained model, etc.

```python
class PSession :
    
    def __init__( self ) :
        # svm model 
        self.model = None
        # input and output converters
        self.inScaler = None
        self.outEncoder = None
        # k-fold cross validator
        self.kf = None
        # scores and predictions
        self.scores = None
        self.predictions = None
        self.accuracyScore = None

        # training data
        self.trainX = None
        self.trainY = None

        # session parameters
        self.nbinsColors = 1
        self.nbinsNormals = 1
        self.kernel = 'linear'
        self.gamma = 1.0
        self.C = 1.0
        self.degree = 1
        self.dataPercent = 0.5
        self.trainSize = 0
```

We defined a training **schedule** for the options we could take, and generated sessions based on this schedule. This can be found in the **startTrainingSchedule** method, which defines all the options for the experiments.

We then run all the sessions and saved the results ( confusion matrices and scores ) for later checking, and also sorted to show the best in the top of the list saved. All models accuracies were obtained by using **5-fold** **crossvalidation**.

The final results can be found in the [**resultsKlinear**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/results/linear/resultsKlinear.txt) and [**resultsKpoly**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/results/poly/resultsKpoly.txt) files. ( The following shows only the top 10 models, the full list can be found in the appropiate files)

```txt
TOP 10 LINEAR MODELS
nc : binsize colors histogram [ 32, 128, 255 ]
nn : binsize normals histogram [ 50, 150, 250 ]
size : number training samples [ 10%, 50%, 100% ] -> [ 1600, 8000, 16000 ]
C : amount of regularization [ 0.1, 1.0, 10.0 ]


sess_kernel_linear_C_0.1_dg_1_gamma_1.0_nc_255_nn_50_size_16000 - 0.959625
sess_kernel_linear_C_0.1_dg_1_gamma_1.0_nc_128_nn_50_size_16000 - 0.9595625
sess_kernel_linear_C_1.0_dg_1_gamma_1.0_nc_128_nn_50_size_16000 - 0.955
sess_kernel_linear_C_0.1_dg_1_gamma_1.0_nc_32_nn_50_size_16000 - 0.9538125
sess_kernel_linear_C_10.0_dg_1_gamma_1.0_nc_128_nn_50_size_16000 - 0.9534375
sess_kernel_linear_C_1.0_dg_1_gamma_1.0_nc_255_nn_50_size_16000 - 0.9524375
sess_kernel_linear_C_1.0_dg_1_gamma_1.0_nc_32_nn_50_size_16000 - 0.9513125
sess_kernel_linear_C_10.0_dg_1_gamma_1.0_nc_255_nn_50_size_16000 - 0.9504375
sess_kernel_linear_C_10.0_dg_1_gamma_1.0_nc_32_nn_50_size_16000 - 0.94875
sess_kernel_linear_C_0.1_dg_1_gamma_1.0_nc_255_nn_150_size_16000 - 0.9454375
```

```txt
TOP 10 POLYNOMIAL MODELS
nc : binsize colors histogram [ 32, 128 ]
nn : binsize normals histogram [ 50, 150 ]
size : number training samples [ 10%, 50%, 100% ] -> [ 1600, 8000, 16000 ]
C : amount of regularization [ 0.1, 1.0, 10.0 ]
dg : degree of the polynomial kernel [ 2, 3, 4 ]

sess_kernel_poly_C_0.1_dg_3_gamma_1.0_nc_128_nn_50_size_16000 - 0.974375
sess_kernel_poly_C_1.0_dg_3_gamma_1.0_nc_128_nn_50_size_16000 - 0.974375
sess_kernel_poly_C_10.0_dg_3_gamma_1.0_nc_128_nn_50_size_16000 - 0.974375
sess_kernel_poly_C_0.1_dg_2_gamma_1.0_nc_128_nn_50_size_16000 - 0.96075
sess_kernel_poly_C_1.0_dg_2_gamma_1.0_nc_128_nn_50_size_16000 - 0.96075
sess_kernel_poly_C_10.0_dg_2_gamma_1.0_nc_128_nn_50_size_16000 - 0.96075
sess_kernel_poly_C_0.1_dg_3_gamma_1.0_nc_128_nn_50_size_8000 - 0.955375
sess_kernel_poly_C_1.0_dg_3_gamma_1.0_nc_128_nn_50_size_8000 - 0.955375
sess_kernel_poly_C_10.0_dg_3_gamma_1.0_nc_128_nn_50_size_8000 - 0.955375
sess_kernel_poly_C_0.1_dg_2_gamma_1.0_nc_128_nn_150_size_16000 - 0.9550625
```
It's worth noting that **we did not included rbf models, as the initial results gave very poor models**. The models overfitted quickly using this type of kernel.

### **_Results_**

From the results of the experiments we decided to pick the best models that have the largest regularization possible ( 10.0 in our experiments ), as they may generalize better. We tested both the **linear** and **polynomial** cases and check if they passed the requirements.

#### **Linear kernel**

For the linear case we used the following model :

*   Bin size for colors histograms : 128
*   Bin size for normals histograms : 50
*   Training size : 100% of the dataset
*   Regularization parameter : 10.0

The resulting confusion matrix after training was the following :

![CONFUSION MATRIX LINEAR MODEL][img_cmatrix_linear]

This is the model that yield the better results when testing in simulation, as it passed the requirements with the following results :

*   **3/3** in **test1.world**
*   **5/5** in **test2.world**
*   **8/8** in **test3.world**

This model gave an accuracy of **95%** in the experiments, and generalized well to our pick-place scenarios. The resulting predicted labels for each scenario are shown in the following figures.

![RESULTS SCENE 1 LINEAR KERNEL][img_results_sc1_klinear]


![RESULTS SCENE 2 LINEAR KERNEL][img_results_sc2_klinear]


![RESULTS SCENE 3 LINEAR KERNEL][img_results_sc3_klinear]

#### **Polynomial kernel**

For the polynomial case we used the following model :

*   Bin size for colors histograms : 128
*   Bin size for normals histograms : 50
*   Training size : 100% of the dataset
*   Regularization parameter : 10.0
*   Degree : 3 ( cubic )

And this is the confusion matrix we got after training :

![CONFUSION MATRIX POLY MODEL][img_cmatrix_poly]

We first tried it with the configuration from before ( same filtering and clustering parameters ) and found that it didn't do a good joob at all, even though its accuracy was of **98%**. We found the problem after playing with the filtering and clustering parameters, changing the leafsize of the downsampling step.

This allowed to have more points for the histograms, which made the features more representative to the model trained, as the model was trained with features obtained from a richer pointcloud ( not downsampled ). Even though normalization was applied to ensure that the histogram has values between 0-1 it still failed to generate good feature vectors. This is why we tweaked the downsampling size by reducing the leafsize of the voxel-grid filter.

The model with the polynomial kernel got an **8/8** in scene 3, which the linear model could not achieve before proper tuning. However, it did not perform as well as the model with the linear kernel in scenes 1 and 2, as it missed one object in each case, so it seems it was not generalizing well enough.

![RESULTS SCENE 1 POLY KERNEL][img_results_sc1_kpoly]


![RESULTS SCENE 2 POLY KERNEL][img_results_sc2_kpoly]


![RESULTS SCENE 3 POLY KERNEL][img_results_sc3_kpoly]

## **Extra-pipeline implementation**

We implemented the extra steps required for the pick-place service call operation. These include creating a collision map for the planner, and making the service call using the already extracted information for the detected objects. In the following subsections we explain each part implemented.

### _**Collision Avoidance**_

In order for the pr2 robot to make a good plan to the object to pick it is necessary to update the collision map used for the planning process, as we don't want our robot to collide with other objects when trying to pick something.

This is achieved by publishing to the **_'/pr2/3d_map/points'_** topic, which is used by moveit to create the collision map.

For some information about these concepts, check [**this**](https://moveit.ros.org/documentation/concepts/) post.

![MOVEIT CONCEPTS][img_moveit_concepts]

Basically, moveit will create an occupancy map from the given pointcloud data coming to that topic, so our only concern is to send the right pointcloud, which should consists of all elements except the current pickable object.

The simpler way we found to achieve this would be to filter out ( with a **cropbox** filter ) the cloud representing the current pickable object from the global scene cloud ( which of course has noise, so it has to be denoised a bit before proceeding ). We **already have these clouds**, as they are the results of the previous steps of the perception pipeline.

![COLLISION MAP GENERATION][img_collisionmap_pipeline]

To achieve this we would need to get the axis aligned bounding box representing the limits of the pickable object pointcloud. This can be easily done similar to the way be obtained the centroid of the pointcloud ( mean ). In our case we would just need to use **max** and **min** to get the appropiate ranges of the bounding box.

Then, we just need to create a cropbox filter with those limits and extract the negative of the filter, which would successfully remove the object from the scene. This functionality can be found in the **_removeFromCloud** method, in the [**PPickPlaceHandler.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PPickPlaceHandler.py) file.

```python
    """
    Removes a given subset cloud from a parent cloud.
    For now, just using cropping based on the boundingbox
    of the subset cloud

    :param parentCloud : cloud to extract the child from - xyz cloud
    :param childCloud : cloud to extract from the parent - xyz cloud
    """
    def _removeFromCloud( self, parentCloud, childCloud ) :
        # compute AABB boundaries of child cloud
        _min, _max = self._computeBoundingBox( childCloud )
        # make the cropping filter
        _cropBoxFilter = parentCloud.make_cropbox()
        _cropBoxFilter.set_Negative( True )
        _cropBoxFilter.set_Min( _min[0], _min[1], _min[2], 1.0 )
        _cropBoxFilter.set_Max( _max[0], _max[1], _max[2], 1.0 )

        return _cropBoxFilter.filter()
```

The following picture shows an example of this functionality in action.

![COLLISION MAP 1][img_collision_map_1]

### _**Robot motion**_

The collision map created only represents the collidable parts in front of the robot, but we would like to have other parts of the scene to also be taken into account ( left and right parts of the table, and the objects in the boxes, if necessary ).

To achieve this, we would need to add to the published pointcloud this parts of the scene, for which we need the robot to rotate in place to see those parts.

This part is implemented in the [**PPickPlaceHandler.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception/PPickPlaceHandler.py), were we basically send requests to the topic **_'/pr2/world_joint_controller/command'_**, which controls the base joint of the robot.

```python
    class PPickPlaceHandler( object ) :

        def __init__( self ) :

            # ...

            # publisher for robot base motion
            self.m_pubBaseMotion = rospy.Publisher( '/pr2/world_joint_controller/command',
                                                    Float64,
                                                    queue_size = 10 )

            # ...

        # ...

        def requestBaseMotion( self, angle ) :
            print 'requesting base motion by angle: ', angle
            self.m_pubBaseMotion.publish( ( angle * np.pi / 180.0 ) )

        # ...

        def _hasReachedReference( self, current, reference ) :
            if abs( current - reference ) < 0.01 :
                return True
            return False

        def makeRightScan( self, worldCloud ) :
            self.requestBaseMotion( 120.0 )
            if self._hasReachedReference( self.m_worldJointAngle, np.radians( 120.0 ) ) :
                # add the current cloud to the collision map
                self.addSideCollisionCloud( worldCloud )
                return True
            return False

        def makeLeftScan( self, worldCloud ) :
            self.requestBaseMotion( -120.0 )
            if self._hasReachedReference( self.m_worldJointAngle, np.radians( -120.0 ) ) :
                # add the current cloud to the collision map
                self.addSideCollisionCloud( worldCloud )
                return True
            return False

        def makeReturnScan( self ) :
            self.requestBaseMotion( 0.0 )
            if self._hasReachedReference( self.m_worldJointAngle, 0.0 ) :
                return True
            return False
```

### _**Pick & place state machine**_

Finally, to combine all these steps we had to implement a system that would allows us to handle all the components we already have. At first, it was easy as we had just the pipeline; but now we have to combine them accordingly.

This is done by adding a high level state machine that uses these different components, which is depicted in the following figure.

![STATE MACHINE][img_statemachine_pick_place]

This implementation can be found in the [**perception_pipeline.py**](https://github.com/wpumacay/RoboND-Perception-Project/blob/master/pr2_robot/scripts/perception_pipeline.py) file, whose class **PPipeline** handles the different pipeline components with this state machine; and the actual operation is shown below.

![PICK PLACE FULL OPERATION][gif_pick_place_full]

The robot follows the state machine and successfully makes a pick & place operation. It also publishes the right collision map to avoid colliding with objects and parts of the scene.

Unfortunately, most than 50% of the pick-place request failed in the **pick** step, as the robot doesn't have a mechanism to ensure it has actually picked the object. This is provided by the service, which waits a little till the gripper closes. This works sometimes, but sometime it doesn't quite grasp the object; and it's way more frequent than the previous kinematics project, which only happened very few times.

![PICK PLACE FULL OPERATION FAIL][gif_pickplace_operation_grasp_fail]

## CONCLUSIONS

After the final implementation we got the required results and got the pick & place operation to work. These are some of the steps we successfully solved with the approach we followed :

*   **An SVM with engineered features** is quite good at solving the task of recognizing the objects in the scene. We had to tweak a bit some parts of the pipeline for it to work correctly, but if done properly it yields good results.

*   **Picking the model** through experiments turned out to be helpful when tuning the parameters of the model. This led us to a model that satisfied the requirements of the project. However, this approach should be replaced by first picking the most important hyperparameters of the model, and doing the search in the space defined by only those, as it would save more time.

*   **The use of high-level handlers** helps to make the pipeline more robust and easy to test. Again, we implemented the pipeline in separated modules, each being tested and tuned separately.

We also found some problems that might arise because of how we approached to solve this problem :

*   **The implementation is very sequential** and some specific requirements are necessary before going to the next step. If some step of the pipeline breaks, then the whole process is wrong. This can be fixed to some extent by making more robust high-level handlers of the low-level pipeline steps.

*   **Problem of dealing with dynamic environments**, as we are updating the collision map assuming it will not change after we made the scan of a certain area.

*   **The pointcloud filtering part** is the most compute-intensive of the whole pipeline. The python-pcl bindings work quite well, but in some parts ( specially with the statistical outlier removal ) it took too much time.

## FUTURE WORK

We think that adding some extras would be great, like :

*   Adding some features to control both arms, or some section as to how to make this possible would be great. Perhaps after a second revision I will try a pull request to add some features to the project

*   Fixing the gripper issue when picking objects, as it was also a bit recurrent in the kinematics project. I'm quite certain that is not an issue with the centroid, as I always checked in the simulator the location of the gripper when picking the objects. It even happened in simulation mode.

*   Perhaps changing the properties of the simulator would help, as the simulation is a bit weird ( some objects don't fall the way they should, or keep floating around a pivot point when they should just fall naturally ). Perhaps the simulation configuration is being set to small quality, or something similar, that might be causing these weird artifacts.
#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

g_model = None
g_clf = None
g_encoder = None
g_scaler = None

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    _pcl_cloud = ros_to_pcl( pcl_msg )

    # Voxel Grid Downsampling
    _voxel_filter = _pcl_cloud.make_voxel_grid_filter()
    _voxel_filter.set_leaf_size( 0.01, 0.01, 0.01 )
    _pcl_cloud = _voxel_filter.filter()
    
    # PassThrough Filter
    _pass_filter = _pcl_cloud.make_passthrough_filter()
    _pass_filter.set_filter_field_name( 'z' )
    _pass_filter.set_filter_limits( 0.6, 1.1 )
    _pcl_cloud = _pass_filter.filter()

    # RANSAC Plane Segmentation
    _ransac_seg = _pcl_cloud.make_segmenter()
    _ransac_seg.set_model_type( pcl.SACMODEL_PLANE )
    _ransac_seg.set_method_type( pcl.SAC_RANSAC )
    _ransac_seg.set_distance_threshold( 0.02 )
    _plane_indices, _ = _ransac_seg.segment()

    # Extract inliers and outliers
    _table_cloud = _pcl_cloud.extract( _plane_indices, negative = False )
    _objects_cloud = _pcl_cloud.extract( _plane_indices, negative = True )

    # Euclidean Clustering

    _xyz_objects_cloud = XYZRGB_to_XYZ( _objects_cloud )
    _kdtree = _xyz_objects_cloud.make_kdtree()
    
    _cluster_extractor = _xyz_objects_cloud.make_EuclideanClusterExtraction()
    _cluster_extractor.set_ClusterTolerance( 0.05 )
    _cluster_extractor.set_MinClusterSize( 30 )
    _cluster_extractor.set_MaxClusterSize( 2000 )
    _cluster_extractor.set_SearchMethod( _kdtree )
    _cluster_indices = _cluster_extractor.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    _cluster_color = get_color_list( len( _cluster_indices ) )
    _color_cluster_point_list = []

    for j, indices in enumerate( _cluster_indices ) :
        for i, index in enumerate( indices ) :
            _color_cluster_point_list.append( [ _xyz_objects_cloud[ index ][0],
                                                _xyz_objects_cloud[ index ][1],
                                                _xyz_objects_cloud[ index ][2],
                                                rgb_to_float( _cluster_color[j] ) ] )

    _cluster_cloud = pcl.PointCloud_PointXYZRGB()
    _cluster_cloud.from_list( _color_cluster_point_list )
    _ros_cloud_cluster = pcl_to_ros( _cluster_cloud )

    # Convert PCL data to ROS messages
    _ros_cloud_objects = pcl_to_ros( _objects_cloud )
    _ros_cloud_table = pcl_to_ros( _table_cloud )

    # Publish ROS messages
    pcl_objects_pub.publish( _ros_cloud_objects )
    pcl_table_pub.publish( _ros_cloud_table )
    pcl_cluster_pub.publish( _ros_cloud_cluster )

    # Exercise-3 *****************************************

    # Classify the clusters! (loop through each detected cluster one at a time)
    _detected_objects_labels = []
    _detected_objects = []

    for _index, _pts_list in enumerate( _cluster_indices ) :

        # Grab the points for the cluster
        _pcl_cluster_cloud = _objects_cloud.extract( _pts_list )

        # Compute the associated feature vector
        _ros_cluster_cloud = pcl_to_ros( _pcl_cluster_cloud )

        _chists = compute_color_histograms( _ros_cluster_cloud )
        _normals = get_normals( _ros_cluster_cloud )
        _nhists = compute_normal_histograms( _normals )
        _feature_vector = np.concatenate( ( _chists, _nhists ) )

        # Make the prediction
        _prediction = g_clf.predict( g_scaler.transform( _feature_vector.reshape( 1, -1 ) ) )
        _label = g_encoder.inverse_transform( _prediction )[0]
        _detected_objects_labels.append( _label )

        # Publish a label into RViz
        _label_pos = list( _xyz_objects_cloud[ _pts_list[0] ] )
        _label_pos[2] += 0.4
        object_markers_pub.publish( make_label( _label, _label_pos, _index ) )

        # Add the detected object to the list of detected objects.
        _do = DetectedObject()
        _do.label = _label
        _do.cloud = _ros_cluster_cloud
        _detected_objects.append( _do )

    rospy.loginfo( 'Detected {} objects: {}'.format( len( _detected_objects_labels ), _detected_objects_labels ) )

    # Publish the list of detected objects
    detected_objects_pub.publish( _detected_objects )

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node( 'clustering', anonymous = True )
    # Create Subscribers
    pcl_sub = rospy.Subscriber( "/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size = 1 )
    # Create Publishers
    pcl_objects_pub = rospy.Publisher( "/pcl_objects", PointCloud2, queue_size = 1 )
    pcl_table_pub = rospy.Publisher( "/pcl_table", PointCloud2, queue_size = 1 )
    pcl_cluster_pub = rospy.Publisher( "/pcl_cluster", PointCloud2, queue_size = 1 )
    object_markers_pub = rospy.Publisher( "/object_markers", Marker, queue_size = 1 )
    detected_objects_pub = rospy.Publisher( "/detected_objects", DetectedObjectsArray, queue_size = 1 )

    g_model = pickle.load( open( 'model.sav', 'rb' ) )
    g_clf = g_model['classifier']
    g_encoder = LabelEncoder()
    g_encoder.classes_ = g_model['classes']
    g_scaler = g_model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown() :
        rospy.spin()

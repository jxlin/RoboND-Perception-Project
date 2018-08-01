#!/usr/bin/env python

# Some libraries needed
import rospy
import numpy as np
import time
import pickle
import sklearn
# sensor stick package functionality
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.srv import GetNormals
# messages and services definitions
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
# import our pipeline-steps functionality
from perception.PCloudFilter import *
from perception.PCloudClusterMaker import *
from perception.PClassifierSVM import *
from perception.PPickPlaceHandler import *
from perception.PUtils import *

g_pipeline = None

class PPipeline( object ) :

    def __init__( self ) :
        super( PPipeline, self ).__init__()
        # initialize resources
        self._createPipeline()
        self._createPublishers()
        self._createSubscribers()

    def _createPipeline( self ) :
        # cloud filterer
        self.m_cloudFilterer = PCloudFilter()
        # cloud clusterer
        self.m_cloudClusterer = PCloudClusterMaker()
        # create classifier and load model from disk
        self.m_classifier = PClassifierSVM()
        self.m_classifier.loadModel( '../data/model_c255_n250_2000.sav' )
        # create pick-place handler
        self.m_pickplaceHandler = PPickPlaceHandler( sceneNum = 2 )

    def _createPublishers( self ) :
        # publishers to send the filtered table and objects to RViz
        self.m_pubVizCloudTable = rospy.Publisher( '/pipeline/filtering/table',
                                                   PointCloud2,
                                                   queue_size = 1 )
        self.m_pubVizCloudObjects = rospy.Publisher( '/pipeline/filtering/objects',
                                                     PointCloud2,
                                                     queue_size = 1 )
        # publisher to send the clusters visualizations to RViz
        self.m_pubVizClusters = rospy.Publisher( '/pipeline/clustering/clusters',
                                                 PointCloud2,
                                                 queue_size = 1 )
        # dummy cloud for some tests
        self.m_pubVizDummyCloud = rospy.Publisher( '/pipeline/dummy',
                                                   PointCloud2,
                                                   queue_size = 1 )
        # publisher to send the markers for the detected objects
        self.m_pubVizObjectsMarkers = rospy.Publisher( '/pipeline/classification/objectMarkers',
                                                       Marker,
                                                       queue_size = 1 )
        # publisher to send the detected objects list
        self.m_pubDetectedObjects = rospy.Publisher( '/detected_objects',
                                                     DetectedObjectsArray,
                                                     queue_size = 1 )

    def _createSubscribers( self ) :
        # subscribe to the rgbd pointcloud topic from the pr2's rgbd camera
        self.m_subsRGBDcloud = rospy.Subscriber( '/pr2/world/points',
                                                 pc2.PointCloud2,
                                                 self.onCloudMessageReceived,
                                                 queue_size = 1 )

    def getNormals( self, cloud ) :
        _getNormalsProxy = rospy.ServiceProxy( '/feature_extractor/get_normals', GetNormals )
        return _getNormalsProxy( cloud ).cluster

    def onCloudMessageReceived( self, cloudMsg ) :
        # transform roscloud to pcl format
        _cloudPcl = ros_to_pcl( cloudMsg )

        # SEGMENTATION AND CLUSTERING ###################################

        # apply segmentation
        # _tableCloud, _objectsCloud = self.m_cloudFilterer.apply( _cloudPcl, self.m_pubVizDummyCloud )
        _tableCloud, _objectsCloud = self.m_cloudFilterer.apply( _cloudPcl )
        # apply clusterer
        _clustersClouds, _clustersCloudViz = self.m_cloudClusterer.cluster( _objectsCloud )
        # transforms results for publishers
        _rosTableCloud = pcl_to_ros( _tableCloud )
        _rosObjectsCloud = pcl_to_ros( _objectsCloud )
        _rosClusterVizCloud = pcl_to_ros( _clustersCloudViz )
        # publish intermediate results
        self.m_pubVizCloudTable.publish( _rosTableCloud )
        self.m_pubVizCloudObjects.publish( _rosObjectsCloud )
        self.m_pubVizClusters.publish( _rosClusterVizCloud )

        # ###############################################################

        # CLASSIFICATION ################################################
        print 'START TIMING - CLASSIFICATION ******************'
        _detectedObjectsLabels = []
        _detectedObjects = []
        _t1 = time.time()
        for i in range( len( _clustersClouds ) ) :
            # get a specific object cloud
            _objCloud = _clustersClouds[i]
            # transform it to ros, as the feature ...
            # extraction works with ros clouds
            _rosObjCloud = pcl_to_ros( _objCloud )
            ## Compute features #########################################
            # colors histograms
            _chists = computeColorHistograms( _rosObjCloud, nbins = 255 )
            # normals histograms
            _normalsCloud = self.getNormals( _rosObjCloud )
            _nhists = computeNormalHistograms( _normalsCloud, nbins = 250 )
            # make the feature vector
            _featureVector = np.concatenate( ( _chists, _nhists ) )
            #############################################################
            # Make the prediction
            _predictedLabel = self.m_classifier.predict( _featureVector.reshape( 1, -1 ) )
            _detectedObjectsLabels.append( _predictedLabel )
            # Publish label to RViz
            _labelPos = list( _clustersClouds[i][0] )
            _labelPos[2] += 0.4
            self.m_pubVizObjectsMarkers.publish( make_label( _predictedLabel, _labelPos, i ) )
            # Add the detected object to the list of detected objects
            _do = DetectedObject()
            _do.label = _predictedLabel
            _do.cloud = _rosObjCloud
            _detectedObjects.append( _do )
        _t2 = time.time()
        print 'classification: ', ( ( _t2 - _t1 ) * 1000 ), ' ms'
        print 'END TIMING - CLASSIFICATION ******************'
        rospy.loginfo( 'Detected {} objects: {}'.format( len( _detectedObjectsLabels ), _detectedObjectsLabels ) )
        # Publish the list of detected objects
        self.m_pubDetectedObjects.publish( _detectedObjects )
        # ###############################################################

        # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
        # Could add some logic to determine whether or not your object detections are robust
        # before calling pr2_mover()
        try:
            self.m_pickplaceHandler.pickObjectsFromList( _detectedObjects, callservice = False, savetofile = False )
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node( 'perception_pipeline', anonymous = True )
    # Initialize color_list
    get_color_list.color_list = []
    # initialize pipeline
    g_pipeline = PPipeline()

    # Spin while node is not shutdown
    while not rospy.is_shutdown() :
        rospy.spin()
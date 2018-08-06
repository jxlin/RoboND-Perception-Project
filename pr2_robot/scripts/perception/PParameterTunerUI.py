#!/usr/bin/env python

############################################################################
# This node implements a simple UI that lets us pick the right ...
# parameters for our filtering and clustering steps.
# The results ( filtered point clouds, and clustered point clouds ) are ...
# published to some topics defined here for easy debugging in rviz.
############################################################################

import os
import sys
sys.path.insert( 0, '../' )
import threading
import numpy as np
import rospy
import time

# binding provider is PyQt5; use its own specific API
# Check here for a good tutorial : http://zetcode.com/gui/pyqt5/
from python_qt_binding.QtWidgets import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *

# to publish regions as markers
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# import filtering and clustering wrappers
from PCloudFilter import *
from PCloudClusterMaker import *
# import utils
from PUtils import *

DEFAULT_RANGE_TICKS = 1000

class SliderWrapper :

    def __init__( self, layout, slider, label, text, value, rmin, rmax, ticks ) :
        # widgets
        self.m_wslider = slider
        self.m_wlabel = label
        self.m_wtext = text
        # layout holder
        self.m_layout = layout
        # properties
        self.m_min = rmin
        self.m_max = rmax
        self.m_range = rmax - rmin
        self.m_ticks = ticks

        self.m_value = 0

        # make slider be at that default value
        self.m_wslider.setValue( ( ( value - self.m_min ) / ( self.m_range ) ) * ( self.m_ticks ) )
        # update internal value
        self._updateValue()

    def getLayout( self ) :
        return self.m_layout

    def getMin( self ) : 
        return self.m_min

    def getMax( self ) : 
        return self.m_max

    def getRange( self ) : 
        return self.m_range

    def getTicks( self ) :
        return self.m_ticks

    def getValue( self ) :
        return self.m_value

    def _updateValue( self ) :
        self.m_value = self.m_min + self.m_range * ( self.m_wslider.value() / float( self.m_ticks ) )
        self.m_wtext.setText( str( self.m_value ) )


class PParameterTunerUI( QWidget ) :


    def __init__( self ) :

        super( PParameterTunerUI, self ).__init__()

        # create the filter object
        self.m_filter = PCloudFilter()
        # create the cluster generator object
        self.m_clusterGen = PCloudClusterMaker()

        # create some necessary resources
        self._makeResources()
        # build the UI
        self._makeUI()

        # initialize ros thread
        self.m_workerRos.start()

    def _makeSubscribers( self ) :
        # subscribe to the rgbd pointcloud from the pr2 rgbd camera
        self.m_subsRGBDCloud = rospy.Subscriber( '/pr2/world/points',
                                                 pc2.PointCloud2, 
                                                 self._onMessageRGBDPointCloud, 
                                                 queue_size = 1 )

    def _makePublishers( self ) :
        # publisher to send the filtered table and objects to
        self.m_pubFilteredCloudTable = rospy.Publisher( '/filtering/table',
                                                        PointCloud2, 
                                                        queue_size = 1 )
        self.m_pubFilteredCloudObjects = rospy.Publisher( '/filtering/objects',
                                                          PointCloud2, 
                                                          queue_size = 1 )
        # publisher to send the cluster visualization to
        self.m_pubClusteredViz = rospy.Publisher( '/clustering/clusterViz',
                                                  PointCloud2,
                                                  queue_size = 1 )

        # publisher to send markers for the cutting regions
        self.m_pubCutRegions = rospy.Publisher( '/clustering/cutRegions',
                                                MarkerArray,
                                                queue_size = 1 )

    def _makeUI( self ) :
        _ui_vbox = QVBoxLayout()

        self.m_sldFilterSORmeanK = self._makeSlider( 5.0, 20.0, 10, 'SORmeanK' )
        self.m_sldFilterSORthresholdScale = self._makeSlider( 0.001, 0.1, 0.05, 'SORthresholdScale' )

        _ui_vbox.addLayout( self.m_sldFilterSORmeanK.getLayout() )
        _ui_vbox.addLayout( self.m_sldFilterSORthresholdScale.getLayout() )

        self.m_sldFilterLeafSize = self._makeSlider( 0.001, 0.1, 0.01, 'LeafSize' )
        self.m_sldFilterCutMinZ = self._makeSlider( -2, 2 , 0.608, 'CutMinZ' )
        self.m_sldFilterCutMaxZ = self._makeSlider( -2, 2 , 0.88, 'CutMaxZ' )
        self.m_sldFilterCutMinY = self._makeSlider( -2, 2 , -0.456, 'CutMinY' )
        self.m_sldFilterCutMaxY = self._makeSlider( -2, 2 , 0.456, 'CutMaxY' )
        self.m_sldFilterRansacThreshold = self._makeSlider( 0.001, 0.1, 0.00545, 'RansacThreshold' )

        _ui_vbox.addLayout( self.m_sldFilterLeafSize.getLayout() )
        _ui_vbox.addLayout( self.m_sldFilterCutMinZ.getLayout() )
        _ui_vbox.addLayout( self.m_sldFilterCutMaxZ.getLayout() )
        _ui_vbox.addLayout( self.m_sldFilterCutMinY.getLayout() )
        _ui_vbox.addLayout( self.m_sldFilterCutMaxY.getLayout() )
        _ui_vbox.addLayout( self.m_sldFilterRansacThreshold.getLayout() )

        self.m_sldClusteringClusterTolerance = self._makeSlider( 0.01, 1, 0.025, 'ClusterTolerance' )
        self.m_sldClusteringMinClusterSize = self._makeSlider( 10.0, 1000.0, 30, 'MinClusterSize' )
        self.m_sldClusteringMaxClusterSize = self._makeSlider( 10.0, 10000.0, 1000, 'MaxClusterSize' )

        _ui_vbox.addLayout( self.m_sldClusteringClusterTolerance.getLayout() )
        _ui_vbox.addLayout( self.m_sldClusteringMinClusterSize.getLayout() )
        _ui_vbox.addLayout( self.m_sldClusteringMaxClusterSize.getLayout() )

        self.setLayout( _ui_vbox )

    def _makeSlider( self, rmin, rmax, value, name, divs = DEFAULT_RANGE_TICKS ) :
        _vbox = QVBoxLayout()
        _hbox = QHBoxLayout()

        _label = QLabel( name )
        _text = QLineEdit()
        _text.setDisabled( True )
        _slider = QSlider( Qt.Horizontal )
        _slider.setRange( 0, divs )
        _slider.setValue( divs / 2 )

        _hbox.addWidget( _label )
        _hbox.addWidget( _text )
        _vbox.addLayout( _hbox )
        _vbox.addWidget( _slider )

        _sliderWrapped = SliderWrapper( _vbox, _slider, _label, _text, 
                                        value, rmin, rmax, divs )

        # Hack-> Do this after to avoid calling the update value the first time
        _slider.valueChanged.connect( self._onSliderValueChanged )

        return _sliderWrapped

    def _onSliderValueChanged( self ) :
        # acquire permission to change the slider values
        self.m_lock.acquire( True )

        if self.m_sldFilterLeafSize :
            self.m_sldFilterLeafSize._updateValue()
        if self.m_sldFilterCutMinZ :
            self.m_sldFilterCutMinZ._updateValue()
        if self.m_sldFilterCutMaxZ :
            self.m_sldFilterCutMaxZ._updateValue()
        if self.m_sldFilterCutMinY :
            self.m_sldFilterCutMinY._updateValue()
        if self.m_sldFilterCutMaxY :
            self.m_sldFilterCutMaxY._updateValue()
        if self.m_sldFilterRansacThreshold :
            self.m_sldFilterRansacThreshold._updateValue()

        if self.m_sldFilterSORmeanK :
            self.m_sldFilterSORmeanK._updateValue()
        if self.m_sldFilterSORthresholdScale :
            self.m_sldFilterSORthresholdScale._updateValue()

        if self.m_sldClusteringClusterTolerance :
            self.m_sldClusteringClusterTolerance._updateValue()
        if self.m_sldClusteringMinClusterSize :
            self.m_sldClusteringMinClusterSize._updateValue()
        if self.m_sldClusteringMaxClusterSize :
            self.m_sldClusteringMaxClusterSize._updateValue()

        # release lock privileges
        self.m_lock.release()

    def _makeResources( self ) :
        # subscribe to necessary topics
        self._makeSubscribers()
        # create necessary topics
        self._makePublishers()
        # make the ros tick-rate at 10hz
        self.m_rate = rospy.Rate( 10 )
        # create a lock to handle threading
        self.m_lock = threading.Lock()
        # make a worker for the ros functionality
        self.m_workerRos = threading.Thread( target = self._workerFcn )

    def _workerFcn( self ) :
        print 'running rospy thread!!!'
        
        while not rospy.is_shutdown() :
            self.m_rate.sleep()

    def _onMessageRGBDPointCloud( self, cloudMsg ) :
        # grab lock for the slideres
        self.m_lock.acquire( True )
        # make safe usage of the slider values
        self._setFilteringClusteringParams()
        # release the lock for the sliders
        self.m_lock.release()

        if self.m_filter and self.m_clusterGen :
            # convert to pcl type
            _pclCloud = ros_to_pcl( cloudMsg )
            # process the cloud
            self._processCloud( _pclCloud )

    """
    Process the given cloud using the parameters set from the sliders

    :param cloud : pcl cloud data structure
    """
    def _processCloud( self, cloud ) :
            # apply filtering *************************************************************
            _tableCloud, _objectsCloud,_sceneCloud = self.m_filter.apply( cloud )
            # apply clustering
            _clustersClouds, _clustersCloudViz = self.m_clusterGen.cluster( _objectsCloud )
            # *****************************************************************************

            ## Publish the results ********************************************************
            # convert clouds to ros format
            _rosTableCloud = pcl_to_ros( _tableCloud )
            _rosObjectsCloud = pcl_to_ros( _objectsCloud )
            _rosClusterVizCloud = pcl_to_ros( _clustersCloudViz )

            # send the results
            self.m_pubFilteredCloudTable.publish( _rosTableCloud )
            self.m_pubFilteredCloudObjects.publish( _rosObjectsCloud )
            self.m_pubClusteredViz.publish( _rosClusterVizCloud )

            # make also helper markers to visualize some stuff
            self._makeAndSendCutRegions()
            # *****************************************************************************

    def _makeAndSendCutRegions( self ) :
        _regionMarkers = MarkerArray()

        _markerCutPlaneMinZ = self._makePlaneMarker( ( 0, 0, PCloudFilterParams.PASSTHROUGH_LIMITS_Z[0] ), color = ( 1, 0, 0 ) )
        _markerCutPlaneMaxZ = self._makePlaneMarker( ( 0, 0, PCloudFilterParams.PASSTHROUGH_LIMITS_Z[1] ), color = ( 0, 1, 0 ) )

        _markerCutPlaneMinX = self._makePlaneMarker( ( 0, PCloudFilterParams.PASSTHROUGH_LIMITS_Y[0], 1 ), color = ( 1, 1, 0 ), axis = 'y' )
        _markerCutPlaneMaxX = self._makePlaneMarker( ( 0, PCloudFilterParams.PASSTHROUGH_LIMITS_Y[1], 1 ), color = ( 0, 1, 1 ), axis = 'y' )

        _regionMarkers.markers = [ _markerCutPlaneMinZ, _markerCutPlaneMaxZ,
                                   _markerCutPlaneMinX, _markerCutPlaneMaxX ]

        _id = 0
        for m in _regionMarkers.markers :
            m.id = _id
            _id += 1

        self.m_pubCutRegions.publish( _regionMarkers )

    def _makePlaneMarker( self, pos, axis = 'z' , dim = ( 2.0, 2.0 ), color = ( 1.0, 0.0, 0.0 ) ) :
        _marker = Marker()
        # set plane quad properties
        _marker.id = 0
        _marker.type = Marker.TRIANGLE_LIST
        _marker.action = Marker.ADD
        _marker.header.frame_id = 'world'
        _marker.scale.x = 1.0
        _marker.scale.y = 1.0
        _marker.scale.z = 1.0
        _marker.color.r = color[0]
        _marker.color.g = color[1]
        _marker.color.b = color[2]
        _marker.color.a = 0.25
        # set plane quad border points
        _quadPoints = []
        if axis == 'z' :
            _quadPoints = [ ( pos[0] + 0.5 * s0 * dim[0], pos[1] + 0.5 * s1 * dim[1], pos[2] ) 
                            for (s0, s1) in [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)] ]
        elif axis == 'y' :
            _quadPoints = [ ( pos[0] + 0.5 * s0 * dim[0], pos[1], pos[2] + 0.5 * s1 * dim[1] ) 
                            for (s0, s1) in [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)] ]
        elif axis == 'x' :
            _quadPoints = [ ( pos[0], pos[1] + 0.5 * s0 * dim[0], pos[2] + 0.5 * s1 * dim[1] ) 
                            for (s0, s1) in [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)] ]
        else :
            print 'No axis set correctly when creating a plane marker visualization'

        _marker.points = [ Point( p[0], p[1], p[2] ) for p in _quadPoints ]

        return _marker

    def _setFilteringClusteringParams( self ) :

        if self.m_sldFilterLeafSize :
            PCloudFilterParams.VOXEL_LEAF_SIZE[0] = self.m_sldFilterLeafSize.getValue()
            PCloudFilterParams.VOXEL_LEAF_SIZE[1] = self.m_sldFilterLeafSize.getValue()
            PCloudFilterParams.VOXEL_LEAF_SIZE[2] = self.m_sldFilterLeafSize.getValue()
        if self.m_sldFilterCutMinZ :
            PCloudFilterParams.PASSTHROUGH_LIMITS_Z[0] = self.m_sldFilterCutMinZ.getValue()
        if self.m_sldFilterCutMaxZ :
            PCloudFilterParams.PASSTHROUGH_LIMITS_Z[1] = self.m_sldFilterCutMaxZ.getValue()
        if self.m_sldFilterCutMinY :
            PCloudFilterParams.PASSTHROUGH_LIMITS_Y[0] = self.m_sldFilterCutMinY.getValue()
        if self.m_sldFilterCutMaxY :
            PCloudFilterParams.PASSTHROUGH_LIMITS_Y[1] = self.m_sldFilterCutMaxY.getValue()
        if self.m_sldFilterRansacThreshold :
            PCloudFilterParams.RANSAC_THRESHOLD = self.m_sldFilterRansacThreshold.getValue()

        if self.m_sldFilterSORmeanK :
            PCloudFilterParams.SOR_MEAN_K = int( self.m_sldFilterSORmeanK.getValue() )
        if self.m_sldFilterSORthresholdScale :
            PCloudFilterParams.SOR_THRESHOLD_SCALE = self.m_sldFilterSORthresholdScale.getValue()

        if self.m_sldClusteringClusterTolerance :
            PCloudClusterMakerParams.DBSCAN_CLUSTER_TOLERANCE = self.m_sldClusteringClusterTolerance.getValue()
        if self.m_sldClusteringMinClusterSize :
            PCloudClusterMakerParams.DBSCAN_MIN_CLUSTER_SIZE = self.m_sldClusteringMinClusterSize.getValue()
        if self.m_sldClusteringMaxClusterSize :
            PCloudClusterMakerParams.DBSCAN_MAX_CLUSTER_SIZE = self.m_sldClusteringMaxClusterSize.getValue()

    def closeEvent( self, event ) :
        # stop rospy manually
        rospy.signal_shutdown( 'Closing UI' )
        # accept and do the default behavior
        event.accept()
        # clean some stuff stuff
        self.m_workerRos.join()

if __name__ == '__main__' :

    rospy.init_node( 'parameterTunerUI', disable_signals = True )

    # Initialize color_list
    get_color_list.color_list = []

    _app = QApplication( sys.argv )

    _ui = PParameterTunerUI()
    _ui.show()

    sys.exit( _app.exec_() )
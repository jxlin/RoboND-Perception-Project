#############################################################################
#   This helper module implements the necessary ...
#   clustering steps to be applied to our filtered point cloud
#   Author: Wilbert Pumacay - a.k.a Daru
#############################################################################

import numpy as np
import pcl
import time

from PUtils import *

class PCloudClusterMakerParams :
    # DBSCAN params
    DBSCAN_CLUSTER_TOLERANCE = 0.025
    DBSCAN_MIN_CLUSTER_SIZE = 30
    DBSCAN_MAX_CLUSTER_SIZE = 2000


class PCloudClusterMaker( object ) :

    def __init__( self ) :
        super( PCloudClusterMaker, self ).__init__()

    def cluster( self, cloud ) :
        # Apply dbscan get the cluster indices
        # print 'START TIMING - CLUSTERING ******************'
        _t1 = time.time()
        _clusterIndices = self._dbscan( cloud )
        _t2 = time.time()
        # print 'dbscan: ', ( ( _t2 - _t1 ) * 1000 ), ' ms'
        # print 'END TIMING - CLUSTERING ********************'
        # Create a cloud with colors for each cluster
        _cloudClustersColored = self._makeCloudForClustersViz( cloud,
                                                               _clusterIndices )
        # Create clouds for each cluster
        _cloudsClusters = self._makeCloudsFromCluster( cloud,
                                                       _clusterIndices )

        return _cloudsClusters, _cloudClustersColored

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

    def _makeCloudForClustersViz( self, cloud, clusterIndices ) :
        # generate some colors for the clusters
        _clusterColors = get_color_list( len( clusterIndices ) )
        # Create a list of the points for the pointcloud
        _cloudPointList = []

        for clusterId, indices in enumerate( clusterIndices ) :
            for _, index in enumerate( indices ) :
                _cloudPointList.append( [ cloud[ index ][ 0 ],
                                          cloud[ index ][ 1 ],
                                          cloud[ index ][ 2 ],
                                          rgb_to_float( _clusterColors[ clusterId ] ) ] )
        # Create the actual viz cloud
        _cloudViz = pcl.PointCloud_PointXYZRGB()
        _cloudViz.from_list( _cloudPointList )

        return _cloudViz

    def _makeCloudsFromCluster( self, cloud, clusterIndices ) :
        _cloudsClusters = []

        for _, indices in enumerate( clusterIndices ) :
            _cloudsClusters.append( cloud.extract( indices ) )

        return _cloudsClusters
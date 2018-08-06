#############################################################################
#   This helper module implements the necessary ...
#   filtering steps to be applied to our point clouds
#   Author: Wilbert Pumacay - a.k.a Daru
#############################################################################

import numpy as np
import pcl
import time

from PUtils import *

class PCloudFilterParams :
    # Voxel grid downsample params
    # VOXEL_LEAF_SIZE = [ 0.01, 0.01, 0.01 ]
    # VOXEL_LEAF_SIZE = [ 0.005, 0.005, 0.005 ]
    VOXEL_LEAF_SIZE = [ 0.0035, 0.0035, 0.0035 ]
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

    

class PCloudFilter ( object ) :

    def __init__( self ) :
        super( PCloudFilter, self ).__init__()

    """
    Applies cloud filtering steps to the given cloud and ...
    returns the table, objects clouds and the combination of the two

    :param cloud : pcl cloud data structure
    :param optpub : optional publisher to send cloud to
    """
    def apply( self, cloud, optpub = None ) :
        # print 'START TIMING - FILTERING ******************'
        _t1 = time.time()
        _fcloud = self._denoise( cloud )
        if optpub is not None :
            optpub.publish( pcl_to_ros( _fcloud ) )
        _t2 = time.time()
        _fcloud = self._voxelGridDownsample( _fcloud )
        _t3 = time.time()
        _fcloud = self._passThroughFiltering( _fcloud )
        _t4 = time.time()
        _tableCloud, _objectsCloud = self._ransacSegmentation( _fcloud )
        _t5 = time.time()

        # print 'denoisingf: ', ( ( _t2 - _t1 ) * 1000 ), ' ms'
        # print 'downsampling: ', ( ( _t3 - _t2 ) * 1000 ), ' ms'
        # print 'passthroughf: ', ( ( _t4 - _t3 ) * 1000 ), ' ms'
        # print 'ransacsegmen: ', ( ( _t5 - _t4 ) * 1000 ), ' ms'

        # print 'END TIMING - FILTERING ********************'

        return _tableCloud, _objectsCloud, _fcloud

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

    """
    Applies statistical outlier removal
    :param cloud : the cloud to remove outliers from
    """
    def _denoise( self, cloud ) :
        _filter = cloud.make_statistical_outlier_filter()
        _filter.set_mean_k( PCloudFilterParams.SOR_MEAN_K )
        _filter.set_std_dev_mul_thresh( PCloudFilterParams.SOR_THRESHOLD_SCALE )

        return _filter.filter()
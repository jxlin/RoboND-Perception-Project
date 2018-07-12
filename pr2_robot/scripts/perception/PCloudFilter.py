
###
# This helper module implements the necessary ...
# filtering steps to be applied to our point clouds
###

import numpy as np
import pcl

class PCloudFilterParams :
    # Voxel grid downsample params
    VOXEL_LEAF_SIZE = [ 0.01, 0.01, 0.01 ]
    # Passthrough filter params
    PASSTHROUGH_AXIS = 'z'
    PASSTHROUGH_LIMITS = [ 0.36, 0.84 ]
    # RANSAC segmentation params
    RANSAC_THRESHOLD = 0.03
    # Statistical Outlier Removal (SOR) params
    SOR_MEAN_K = 50
    SOR_THRESHOLD_SCALE = 1.0

    

class PCloudFilter ( object ) :

    def __init__( self ) :
        super( PCloudFilter, self ).__init__()

    """
    Applies cloud filtering steps to the given cloud and ...
    returns the table and objects clouds

    :param cloud : pcl cloud data structure
    """
    def apply( self, cloud ) :

        _fcloud = self._voxelGridDownsample( cloud )
        _fcloud = self._passThroughFiltering( _fcloud )
        _tableCloud, _objectsCloud = self._ransacSegmentation( _fcloud )

        return _tableCloud, _objectsCloud

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
        _filter = cloud.make_passthrough_filter()
        _filter.set_filter_field_name( PCloudFilterParams.PASSTHROUGH_AXIS )
        _filter.set_filter_limits( PCloudFilterParams.PASSTHROUGH_LIMITS[0],
                                   PCloudFilterParams.PASSTHROUGH_LIMITS[1] )

        return _filter.filter()

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
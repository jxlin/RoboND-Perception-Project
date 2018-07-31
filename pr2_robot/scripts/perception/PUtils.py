#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# Import modules
import rospy
import pcl
import numpy as np
import math
import ctypes
import struct
import sensor_msgs.point_cloud2 as pc2
import matplotlib.colors
import matplotlib.pyplot as plt

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from random import randint
from rospy_message_converter import message_converter
import yaml

def random_color_gen():
    """ Generates a random color

        Args: None

        Returns:
            list: 3 elements, R, G, and B
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]


def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2], data[3]])

    pcl_data = pcl.PointCloud_PointXYZRGB()
    pcl_data.from_list(points_list)

    return pcl_data


def pcl_to_ros(pcl_array):
    """ Converts a pcl PointXYZRGB to a ROS PointCloud2 message

        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud

        Returns:
            PointCloud2: A ROS point cloud
    """
    ros_msg = PointCloud2()

    ros_msg.header.stamp = rospy.Time.now()
    ros_msg.header.frame_id = "world"

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="rgb",
                            offset=16,
                            datatype=PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width * ros_msg.height
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))

    ros_msg.data = "".join(buffer)

    return ros_msg


def XYZRGB_to_XYZ(XYZRGB_cloud):
    """ Converts a PCL XYZRGB point cloud to an XYZ point cloud (removes color info)

        Args:
            XYZRGB_cloud (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud

        Returns:
            PointCloud_PointXYZ: A PCL XYZ point cloud
    """
    XYZ_cloud = pcl.PointCloud()
    points_list = []

    for data in XYZRGB_cloud:
        points_list.append([data[0], data[1], data[2]])

    XYZ_cloud.from_list(points_list)
    return XYZ_cloud


def XYZ_to_XYZRGB(XYZ_cloud, color):
    """ Converts a PCL XYZ point cloud to a PCL XYZRGB point cloud

        All returned points in the XYZRGB cloud will be the color indicated
        by the color parameter.

        Args:
            XYZ_cloud (PointCloud_XYZ): A PCL XYZ point cloud
            color (list): 3-element list of integers [0-255,0-255,0-255]

        Returns:
            PointCloud_PointXYZRGB: A PCL XYZRGB point cloud
    """
    XYZRGB_cloud = pcl.PointCloud_PointXYZRGB()
    points_list = []

    float_rgb = rgb_to_float(color)

    for data in XYZ_cloud:
        points_list.append([data[0], data[1], data[2], float_rgb])

    XYZRGB_cloud.from_list(points_list)
    return XYZRGB_cloud


def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL

        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"

        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]

        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb


def float_to_rgb(float_rgb):
    """ Converts a packed float RGB format to an RGB list

        Args:
            float_rgb: RGB value packed as a float

        Returns:
            color (list): 3-element list of integers [0-255,0-255,0-255]
    """
    s = struct.pack('>f', float_rgb)
    i = struct.unpack('>l', s)[0]
    pack = ctypes.c_uint32(i).value

    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)

    color = [r,g,b]

    return color


def get_color_list(cluster_count):
    """ Returns a list of randomized colors

        Args:
            cluster_count (int): Number of random colors to generate

        Returns:
            (list): List containing 3-element color lists
    """
    if (cluster_count > len(get_color_list.color_list)):
        for i in xrange(len(get_color_list.color_list), cluster_count):
            get_color_list.color_list.append(random_color_gen())
    return get_color_list.color_list

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

#############################################################################
#   Some extra helper functions needed for the perception pipeline
#   Author: Wilbert Pumacay - a.k.a Daru
#############################################################################

"""
Transforms one histogram to another

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

"""
Plots a histogram returned from numpy.histogram
Adapted from this post: https://stackoverflow.com/questions/5328556/histogram-matplotlib

: param hist : numpy histogram
: param rmin : min range for the values of the histogram
: param rmax : max range for the values of the histogram
"""
def plotHistogram( hist, rmin, rmax ) :
    _nbins = len( hist )
    _bins = np.linspace( rmin, rmax, num = ( _nbins + 1 ), endpoint = True )
    _widths = np.diff( _bins )
    _centers = ( _bins[:-1] + _bins[1:] ) / 2.0
    
    plt.figure()
    plt.bar( _centers, hist, align = 'center', width = _widths )
    plt.xticks( _bins )

"""
Normalizes a histogram to have cumsum = 1 ( percentages instead of frequencies )

: param hist : histogram to normalize
"""
def normalizeHistogram( hist ) :
    return hist / float( np.sum( hist ) )

#############################################################################
#   Some extra helper functions needed for the perception pipeline
#   Author: Wilbert Pumacay - a.k.a Daru
#############################################################################

"""
Converts a list of rgb values to a list of hsv values

: param rgbList : rgb list ( 0 - 255 ) to convert to hsv
"""
def rgb2hsv( rgbList ) :
    _rgbNormalized = [ 1.0 * rgbList[0] / 255, 
                       1.0 * rgbList[1] / 255, 
                       1.0 * rgbList[2] / 255 ]
    _hsvNormalized = matplotlib.colors.rgb_to_hsv( [ [ _rgbNormalized ] ] )[0][0]
    return _hsvNormalized

"""
Computes a normalized feature vector ...
from the histograms of the buffers in buffer_list

:param buffer_list: a list of the buffers to use for the histograms
:param nbins: number of bins to generate the histograms
"""
def _featuresFromBuffers( buffer_list, nbins, ranges ) :
    # compute histograms
    _hists = []
    for _buffer in buffer_list :
        _hist, _ = np.histogram( _buffer, bins = nbins, range = ranges )
        _hists.append( _hist )
    
    # concatenate into single feature vector
    _featureVector = np.concatenate( _hists ).astype( np.float64 )

    # normalize feature vector
    _normalizedFeatureVector = _featureVector / np.sum( _featureVector )

    return _normalizedFeatureVector

"""
Computes a feature vector from the color histograms of the given cloud

:param cloud : ros cloud with color information on it
:param using_hsv : flag to whether or not to use hsv colorspace instead
:param nbins : number of bins to use as the size of the histogram
"""
def computeColorHistograms(cloud, using_hsv=True, nbins = 255):
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points( cloud, skip_nans = True ) :
        rgb_list = float_to_rgb( point[3] )
        if using_hsv :
            point_colors_list.append( rgb2hsv( rgb_list ) * 255 )
        else :
            point_colors_list.append( rgb_list )

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append( color[0] )
        channel_2_vals.append( color[1] )
        channel_3_vals.append( color[2] )
    
    # Compute feature vector - use 0 to 255 as range
    normed_features = _featuresFromBuffers( [ channel_1_vals, channel_2_vals, channel_3_vals ], nbins, ( 0., 255. ) )

    return normed_features 

"""
Computes a feature vector from the normals histograms of the given cloud

:param cloud : ros cloud with normals information on it
:param nbins : number of bins to use as the size of the histogram
"""
def computeNormalHistograms( normal_cloud, nbins = 250 ):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points( normal_cloud,
                                           field_names = ( 'normal_x', 'normal_y', 'normal_z' ),
                                           skip_nans = True ):
        norm_x_vals.append( norm_component[0] )
        norm_y_vals.append( norm_component[1] )
        norm_z_vals.append( norm_component[2] )

    # Compute feature vector - use -1 to 1 as range
    normed_features = _featuresFromBuffers( [ norm_x_vals, norm_y_vals, norm_z_vals ], nbins, ( -1., 1. ) )

    return normed_features

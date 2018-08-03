#############################################################################
#   This helper module implements the necessary ...
#   functionality to select what to pick for the pick-place routine
#   Author: Wilbert Pumacay - a.k.a Daru
#############################################################################

import numpy as np
import rospy
import pcl
import time

# some necessary messages for the request
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from pr2_robot.srv import *
from std_srvs.srv import *
# some utils we will need
from PUtils import *

def comparatorDistance( obj ) :
    _dx = obj.centroid.position.x
    _dy = obj.centroid.position.y
    _dz = obj.centroid.position.z
    return np.sqrt( _dx ** 2 + _dy ** 2 + _dz ** 2 )

class PPickObject( object ) :

    def __init__( self ) :
        super( PPickObject, self ).__init__()
        # properties
        self.label = ''
        self.group = ''
        self.picked = False
        self.cloud = None
        self.centroid = None

class PPickPlaceHandler( object ) :

    def __init__( self, sceneNum = 1 ) :
        super( PPickPlaceHandler, self ).__init__()
        # define scene number
        self.m_sceneNum = sceneNum
        # object to pick ( type: DetectedObject )
        self.m_detectedObject = None
        # current world-joint angle
        self.m_worldJointAngle = 0.0
        # publisher for robot base motion
        self.m_pubBaseMotion = rospy.Publisher( '/pr2/world_joint_controller/command',
                                                Float64,
                                                queue_size = 10 )
        # publisher for collision pointcloud
        self.m_pubCollision = rospy.Publisher( '/pr2/3d_map/points',
                                               PointCloud2,
                                               queue_size = 1 )
        # subscriber for the robot joint states
        self.m_subsJoints = rospy.Subscriber( '/joint_states',
                                              JointState,
                                              self.onMessageJointsReceived,
                                              queue_size = 1 )
        # list of picked objects
        self.m_pickedList = []
        # initialize handler
        self._initialize()

    def _initialize( self ) :
        # load parameters from parameter server
        _rosPickList = rospy.get_param( '/object_list' )
        # initialize pick list
        self.m_pickDict = {}
        for i in range( len( _rosPickList ) ) :
            _pobject = PPickObject()
            _pobject.label = _rosPickList[i]['name']
            _pobject.group = _rosPickList[i]['group']
            _pobject.picked = False
            _pobject.cloud = None
            _pobject.centroid = None

            self.m_pickDict[ _pobject.label ] = _pobject

        # initialize drop list
        _rosDropList = rospy.get_param( '/dropbox' )
        # initialize drop list
        self.m_dropdict = {}
        for i in range( len( _rosDropList ) ) :
            self.m_dropdict[ _rosDropList[i]['group'] ] = {}
            self.m_dropdict[ _rosDropList[i]['group'] ]['position'] = _rosDropList[i]['position']
            self.m_dropdict[ _rosDropList[i]['group'] ]['name'] = _rosDropList[i]['name']

    """
    Checks whether or not there is a pickable ...
    object in a given list of detected objects

    :param detectecObjects : list of detected objects to check
    """
    def checkSinglePick( self, detectedObjects ) :
        _found = False
        for dobj in detectedObjects :
            # check if in picklist and not picked yet
            if ( dobj.label in self.m_pickDict ) and \
               ( dobj.label not in self.m_pickedList ):
               _found = True
               break

        return _found

    def startScanningPickingProcess( self, detectedObjects, sceneCloud ) :
        # select a single object to pick
        _dobj = self.pickSingleObject( detectedObjects, sceneCloud, False )
        # store this object for later pick-request
        self.m_detectedObject = _dobj
        # make the collision cloud for the tabletop scene, without the detected object
        _collisionCloud = self._removeFromCloud( XYZRGB_to_XYZ( sceneCloud ), 
                                                 XYZRGB_to_XYZ( ros_to_pcl( _dobj.cloud ) ) )
        # clear the octomap
        self._clearOctomap()
        # publish the cloud so far
        self.m_pubCollision.publish( pcl_to_ros( XYZ_to_XYZRGB( _collisionCloud, [255, 0, 0] ) ) )
    
    """
    Continue the picking process using the previous detected object
    """
    def pickCurrentObject( self ) :
        if self.m_detectedObject :
            _yamldict, _response = self._makeSinglePick( self.m_detectedObject, None )

    """
    Picks a single object from the requested pick list.

    :param detectedObjects : list of detected objects to pick from
    :param sceneCloud : cloud that represents the tabletop scene, filtered and downsampled
    :param pick : boolean to check whether or not to make the pick request. If false, return the object to pick
    """
    def pickSingleObject( self, detectedObjects, sceneCloud, pick = True ) :
        # compute centroid of the detected objects
        for i in range( len( detectedObjects ) ) :
            _centroid = self._computeCentroid( detectedObjects[i].cloud )
            detectedObjects[i].centroid.position.x = _centroid[0]
            detectedObjects[i].centroid.position.y = _centroid[1]
            detectedObjects[i].centroid.position.z = _centroid[2]
        # order the objects from distance to the camera
        detectedObjects.sort( key = comparatorDistance, reverse = False )
        # find the closest one that has not been picked yet
        for i in range( len( detectedObjects ) ) :
            if detectedObjects[i].label not in self.m_pickedList :
                # try to pick this object
                if pick :
                    self._makeSinglePick( detectedObjects[i], sceneCloud )
                    return None
                else :
                    return detectedObjects[i]

    def _makeSinglePick( self, pickobj, sceneCloud = None ) :
        if sceneCloud :
            # make the appropiate cloud for the collision map
            _collisionCloud = self._removeFromCloud( XYZRGB_to_XYZ( sceneCloud ), 
                                                     XYZRGB_to_XYZ( ros_to_pcl( pickobj.cloud ) ) )
            print 'Cleaning octomap'
            self._clearOctomap()
            print 'Publishing collision cloud'
            self.m_pubCollision.publish( pcl_to_ros( XYZ_to_XYZRGB( _collisionCloud, [255, 0, 0] ) ) )
            # # sleep a little?
            print 'waiting a bit'
            time.sleep( 0.5 )
        
        # set the object properties in the picklist history
        self.m_pickDict[ pickobj.label ].cloud = pickobj.cloud
        self.m_pickDict[ pickobj.label ].picked = True
        self.m_pickDict[ pickobj.label ].centroid = [ pickobj.centroid.position.x,
                                                      pickobj.centroid.position.y,
                                                      pickobj.centroid.position.z ]
        # send the request
        _yamldict, _resp = self._pickObject( self.m_pickDict[ pickobj.label ], True )
        # add it to the picked list
        self.m_pickedList.append( pickobj.label )

        return _yamldict, _resp
        
    """
    Makes the pr2 pick all the objects ...
    in the detected list ( if not picked yet )
    This is kind of an old method. Should use 
    pickSingleObject instead to account for challenge requirements
    """
    def pickObjectsFromList( self, objectList, callservice = False, savetofile = True ) :
        _dicts = []
        # check which object should be picked
        for _keylabel in self.m_pickDict :
            # check if the requested object is not already picked
            if self.m_pickDict[_keylabel].picked :
                continue

            for j in range( len( objectList ) ) :
                # check if the requested object is in the list
                if objectList[j].label != _keylabel :
                    continue
                # set the data to this object
                self.m_pickDict[_keylabel].cloud = objectList[j].cloud
                self.m_pickDict[_keylabel].picked = True
                self.m_pickDict[_keylabel].centroid = self._computeCentroid( objectList[j].cloud )
                # pick the object using the service
                _yamlDict, _ = self._pickObject( self.m_pickDict[_keylabel], callservice = callservice )
                _dicts.append( _yamlDict )

        if savetofile and ( len( _dicts ) > 0 ) :
            send_to_yaml( 'output' + str( self.m_sceneNum ) + '.yaml', _dicts )

    def _computeCentroid( self, cloud ) :
        # ros to pcl conversion
        _points = ros_to_pcl( cloud ).to_array()
        # compute centroid
        _npcentroid = np.mean( _points, axis=0 )[:3]
        # convert it to python's float type
        _centroid = [ np.asscalar( _npcentroid[i] ) for i in range( len( _npcentroid ) ) ]
        return _centroid

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
        ## start pick and place request ##########################################################

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

        #########################################################################################

        return _yamlDict, None

    """
    Clears the octomap of the moveit! planning framework
    """
    def _clearOctomap( self ) :
        rospy.wait_for_service( 'clear_octomap' )
        try :
            _clear_octomap_routine = rospy.ServiceProxy( 'clear_octomap', Empty )
            _clear_octomap_routine()
            
        except rospy.ServiceException, e :
            print 'Failed clearing the octomap: %s' %e

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

    """
    Computes the AABB boundaries of a pointcloud

    :param cloud: pcl cloud with only xyz data
    """
    def _computeBoundingBox( self, cloud ) :
        # transform to points array
        _points = cloud.to_array()
        # compute min-max
        _min = np.min( _points, axis = 0 )
        _max = np.max( _points, axis = 0 )

        return _min, _max

    """
    Make the pr2 base joint go to the given reference

    :param angle : requested reference angle for the base joint
    """
    def requestBaseMotion( self, angle ) :
        print 'requesting base motion by angle: ', angle
        self.m_pubBaseMotion.publish( ( angle * np.pi / 180.0 ) )

    """
    Make the pr2 do a scan of the surroundings
    """
    def requestInitialScan( self ) :
        # to make things simpler, don't check if got ...
        # there, but just delay a bit ( tune delay )

        # go to the side
        print 'requesting rotate left'
        self.requestBaseMotion( -110.0 )
        time.sleep( 25 )
        # go to the other side
        print 'requesting rotate right'
        self.requestBaseMotion( 110.0 )
        time.sleep( 50 )
        # go to neutral position
        print 'requesting back to position'
        self.requestBaseMotion( 0.0 )
        time.sleep( 25 )

    def onMessageJointsReceived( self, jointsMsg ) :
        # get last joint value
        self.m_worldJointAngle = jointsMsg.position[-1]
        # print 'current worldjoint angle: ', self.m_worldJointAngle

    def _hasReachedReference( self, current, reference ) :
        if abs( current - reference ) < 0.01 :
            return True
        return False

    def makeRightScan( self, worldCloud ) :
        self.requestBaseMotion( 110.0 )
        if self._hasReachedReference( self.m_worldJointAngle, np.radians( 110.0 ) ) :
            # add the current cloud to the collision map
            self.addSideCollisionCloud( worldCloud )
            return True
        return False

    def makeLeftScan( self, worldCloud ) :
        self.requestBaseMotion( -110.0 )
        if self._hasReachedReference( self.m_worldJointAngle, np.radians( -110.0 ) ) :
            # add the current cloud to the collision map
            self.addSideCollisionCloud( worldCloud )
            return True
        return False

    def makeReturnScan( self ) :
        self.requestBaseMotion( 0.0 )
        if self._hasReachedReference( self.m_worldJointAngle, 0.0 ) :
            return True
        return False

    """
    Denoises and publishes the cloud for the octomap creation

    :param cloud : pcl cloud to send
    """
    def addSideCollisionCloud( self, cloud ) :
        # create the SOR filter
        _filter = cloud.make_statistical_outlier_filter()
        _filter.set_mean_k( 5 )
        _filter.set_std_dev_mul_thresh( 0.001 )
        # denoise the cloud
        _fcloud = _filter.filter()
        # semd it for collision map
        self.m_pubCollision.publish( pcl_to_ros( _fcloud ) )
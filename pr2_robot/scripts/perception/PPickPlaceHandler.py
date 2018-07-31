#############################################################################
#   This helper module implements the necessary ...
#   functionality to select what to pick for the pick-place routine
#   Author: Wilbert Pumacay - a.k.a Daru
#############################################################################

import numpy as np
import rospy
import pcl

# some necessary messages for the request
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
# some utils we will need
from PUtils import *

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
        # initialize handler
        self._initialize()

    def _initialize( self ) :
        # load parameters from parameter server
        _rosPickList = rospy.get_param( '/object_list' )
        # initialize pick list
        self.m_picklist = []
        for i in range( len( _rosPickList ) ) :
            _pobject = PPickObject()
            _pobject.label = _rosPickList[i]['name']
            _pobject.group = _rosPickList[i]['group']
            _pobject.picked = False
            _pobject.cloud = None
            _pobject.centroid = None

            self.m_picklist.append( _pobject )

        # initialize drop list
        _rosDropList = rospy.get_param( '/dropbox' )
        # initialize drop list
        self.m_dropdict = {}
        for i in range( len( _rosDropList ) ) :
            self.m_dropdict[ _rosDropList[i]['group'] ] = {}
            self.m_dropdict[ _rosDropList[i]['group'] ]['position'] = _rosDropList[i]['position']
            self.m_dropdict[ _rosDropList[i]['group'] ]['name'] = _rosDropList[i]['name']
        

    def pickObjectsFromList( self, objectList, callservice = False, savetofile = True ) :
        _dicts = []
        # check which object should be picked
        for i in range( len( self.m_picklist ) ) :
            # check if the requested object is not already picked
            if self.m_picklist[i].picked :
                continue

            for j in range( len( objectList ) ) :
                # check if the requested object is in the list
                if objectList[j].label != self.m_picklist[i].label :
                    continue
                # set the data to this object
                self.m_picklist[i].cloud = objectList[j].cloud
                self.m_picklist[i].picked = True
                self.m_picklist[i].centroid = self._computeCentroid( objectList[j].cloud )
                # pick the object using the service
                _yamlDict = self._pickObject( self.m_picklist[i], callservice = callservice )
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
        ## Send pick and place request ##########################################################

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

        #########################################################################################

        return _yamlDict
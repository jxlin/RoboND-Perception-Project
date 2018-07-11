
import os
import sys
import threading
import numpy as np
import rospy

# binding provider is PyQt5; use its own specific API
# Check here for a good tutorial : http://zetcode.com/gui/pyqt5/
from python_qt_binding.QtWidgets import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *


class PParameterTunerUI( QWidget ) :


    def __init__( self ) :

        super( PParameterTunerUI, self ).__init__()




if __name__ == '__main__' :

    rospy.init_node( 'parameterTunerUI', disable_signals = True )

    _app = QApplication( sys.argv )

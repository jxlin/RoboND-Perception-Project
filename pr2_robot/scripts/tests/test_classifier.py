import sys
sys.path.insert( 0, '../perception/' )

from PClassifier import *

_classifier = PClassifierSVM()
_classifier.startTrainingSchedule( '../../data/training_set_1000.sav' )
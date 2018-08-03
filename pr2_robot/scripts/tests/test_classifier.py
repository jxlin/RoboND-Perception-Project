import sys
sys.path.insert( 0, '../' )

from perception.PClassifierSVM import *

_classifier = PClassifierSVM()
# _classifier.startTrainingSchedule( '../../data/training_set_1000.sav' )
_classifier.train( '../../data/training_set_2000_rm.sav', 'model_c255_n250_2000_rm' )
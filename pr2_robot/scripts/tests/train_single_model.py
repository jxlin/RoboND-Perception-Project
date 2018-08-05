import sys
sys.path.insert( 0, '../' )

from perception.PClassifierSVM import *

# DATASET_FILE = '../../data/training_set_2000.sav'
# MODEL_NAME = 'model_klinear_c128_n50_sz2000_C10'

DATASET_FILE = '../../data/training_set_2000.sav'
MODEL_NAME = 'model_kpoly_deg3_c128_n50_sz2000_C10'

_classifier = PClassifierSVM()
_classifier.train( DATASET_FILE, 
                   MODEL_NAME )
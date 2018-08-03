import sys
sys.path.insert( 0, '../' )

from perception.PClassifierSVM import *

def main( dataset, schedule, experimentName ) :
    _classifier = PClassifierSVM()
    _classifier.startTrainingSchedule( dataset, schedule, experimentName, False )
    # _classifier.startTrainingSchedule( '../../data/training_set_2000.sav' )
    # _classifier.train( '../../data/training_set_2000_rm.sav', 'model_c255_n250_2000_rm' )

if __name__ == '__main__' :
    if len( sys.argv ) != 4 :
        print 'Usage> python test_classifier.py DATASET_SAV_PATH SCHEDULE(NORMAL,KRBF) EXPERIMENT_NAME'
        sys.exit()
    _dataset = sys.argv[1]
    _schedule = sys.argv[2]
    _experimentName = sys.argv[3]

    print 'START'
    main( _dataset, _schedule, _experimentName )
    print 'DONE'
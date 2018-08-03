
echo "STARTED TRAINING SESSION"
# echo "session for NORMAL schedule"
# python test_classifier.py ../../data/training_set_2000.sav NORMAL resultsKlinear
echo "session for KRBF schedule"
python test_classifier.py ../../data/training_set_2000.sav KRBF resultsKrbf
echo "session for KPOLY schedule"
python test_classifier.py ../../data/training_set_2000.sav KPOLY resultsKpoly


import sys
sys.path.insert( 0, '../' )

import numpy as np
import pickle
from perception.PUtils import *

COLOR_HIST_BINS = 255
NORMAL_HIST_BINS = 250
NUM_CLASSES = 8
# 0 - sticky notes
# 1 - book
# 2 - snacks
# 3 - biscuits
# 4 - eraser
# 5 - soap2
# 6 - soap
# 7 - glue
INDEX_START = 0
# target binning
nbinsColors = 64
nbinsNormals = 50

# CHANNELS_COLORS = ['Hue', 'Saturation', 'Value']
# CHANNELS_NORMALS = ['X', 'Y', 'Z']

CHANNELS_COLORS = ['Hue']
CHANNELS_NORMALS = ['X']

TIMEOUT = 3

def rearrangeDataset( dataSet ) :
    # split dataset into samples per class
    _dataSetPerClasses = np.array_split( dataSet, NUM_CLASSES )
    _rdataSet = []
    # rearrange the dataset to have interlaced samples
    for i in range( len( _dataSetPerClasses[0] ) ) :
        for j in range( NUM_CLASSES ) :
            _rdataSet.append( _dataSetPerClasses[j][i] )

    return _rdataSet

def splitFeaturesLabels( dataSet ) :
    # extract data into features and labels
    _dataFeatures = []
    _dataLabels = []
    for _sample in dataSet :
        if np.isnan( _sample[0] ).sum() < 1 :
            _dataFeatures.append( _sample[0] )
            _dataLabels.append( _sample[1] )

    assert ( len( _dataFeatures ) == len( _dataLabels ) ), 'ERROR: features-labels len mismatch'

    return _dataFeatures, _dataLabels

def showFeatures( x, y, ncolors, nnormals, showcolors = True, shownormals = False ) :
    # extract histograms
    _x_chist = x[0:(3 * COLOR_HIST_BINS)]
    _x_nhist = x[(3 * COLOR_HIST_BINS):]
    _x_chist_channels = np.array_split( _x_chist, 3 )
    _x_nhist_channels = np.array_split( _x_nhist, 3 )

    # convert each histogram channel to the desired sizes
    _sx_chist_channels = [ hist2hist( _x_chist_channels[i], ncolors )
                        for i in range( len( _x_chist_channels ) ) ]
    _sx_nhist_channels = [ hist2hist( _x_nhist_channels[i], nnormals )
                        for i in range( len( _x_nhist_channels ) ) ]

    if showcolors :
        # show the original histograms
        for i in range( len( CHANNELS_COLORS ) ) :
            _chist = _x_chist_channels[i]
            _chname = CHANNELS_COLORS[i]
            plotHistogram( _chist, 0, 255, y + ' - COLOR HIST - ' + _chname )
        # show the modified histograms
        for i in range( len( CHANNELS_COLORS ) ) :
            _chist = _sx_chist_channels[i]
            _chname = CHANNELS_COLORS[i]
            plotHistogram( _chist, 0, 255, y + ' - SCALED COLOR HIST - ' + _chname )

    if shownormals :
        # show the original histograms
        for i in range( len( CHANNELS_NORMALS ) ) :
            _nhist = _x_nhist_channels[i]
            _chname = CHANNELS_NORMALS[i]
            plotHistogram( _nhist, -1, 1, y + ' - NORMAL HIST - ' + _chname )
        # show the modified histograms
        for i in range( len( CHANNELS_NORMALS ) ) :
            _nhist = _sx_nhist_channels[i]
            _chname = CHANNELS_NORMALS[i]
            plotHistogram( _nhist, -1, 1, y + ' - SCALED NORMAL HIST - ' + _chname )

def main() :
    # load dataset
    _dataSet = pickle.load( open( '../../data/samples/training_set_2000.sav', 'rb' ) )
    _dataSet = rearrangeDataset( _dataSet )
    _sessionDataX, _sessionDataY = splitFeaturesLabels( _dataSet )
    
    _sampleIndx = 3
    plt.ion()

    while True :
        _x = _sessionDataX[INDEX_START + _sampleIndx * NUM_CLASSES]
        _y = _sessionDataY[INDEX_START + _sampleIndx * NUM_CLASSES]

        print 'sample-x'
        print _x
        print 'sample-y'
        print _y

        # show the current sample of that type
        showFeatures( _x, _y, nbinsColors, nbinsNormals, showcolors = False, shownormals = True )

        _key = plt.waitforbuttonpress( timeout = TIMEOUT )

        print 'key: ', _key
        _sampleIndx += 1

        # close all current figures
        plt.close( 'all' )

if __name__ == '__main__' :
    print 'STARTED FEATURES TEST'
    main()
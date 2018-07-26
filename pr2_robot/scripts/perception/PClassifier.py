
###
# This helper module implements the classifier ...
# of the perception pipeline
###

import sys
import numpy as np
import math
import pcl
import pickle
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from PUtils import *

from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import cross_validation
from sklearn import metrics

# Bins used when taking the huge dataset
DATASET_COLOR_HIST_BINS = 255
DATASET_NORMAL_HIST_BINS = 250
DATASET_NUM_CLASSES = 8

class PSession :
    
    def __init__( self ) :
        # svm model 
        self.model = None
        # input and output converters
        self.inScaler = None
        self.outEncoder = None
        # k-fold cross validator
        self.kf = None
        # scores and predictions
        self.scores = None
        self.predictions = None
        self.accuracyScore = None

        # training data
        self.trainX = None
        self.trainY = None

        # session parameters
        self.nbinsColors = 1
        self.nbinsNormals = 1
        self.kernel = 'linear'
        self.gamma = 1.0
        self.C = 1.0
        self.dataPercent = 0.5
        self.trainSize = 0


class PClassifierSVM( object ) :


    def __init__( self ) :
        super( PClassifierSVM, self ).__init__()
        
        # scaler for the inputs > zero mean and unit variance
        self.m_clfScaler = None
        # actual svm model
        self.m_clfModel = None
        # classes
        self.m_clfLabelEncoder = None

        # data to be used
        self.m_sessionDataX = None
        self.m_sessionDataY = None

    """
    Start a training-tuning session. It handles this by ...
    running the SVM classifier into variations of the ...
    available hyperparameters : 
        - nbins_colors
        - nbins_normals
        - kernel
        - C
        - gamma
        - data_percent

    : param datafile : .sav file that contains the training data
    """
    def startTrainingSchedule( self, datafile ) :
        plt.ion()
        # load the pickle data
        _dataSet = pickle.load( open( datafile, 'rb' ) )
        # rearrange the dataset to have interlaced samples
        _dataSetPerClasses = np.array_split( _dataSet, DATASET_NUM_CLASSES )
        _dataSet = []
        for i in range( len( _dataSetPerClasses[0] ) ) :
            for j in range( DATASET_NUM_CLASSES ) :
                _dataSet.append( _dataSetPerClasses[j][i] )
        # extract data into features and labels
        _dataFeatures = []
        _dataLabels = []
        for _sample in _dataSet :
            if np.isnan( _sample[0] ).sum() < 1 :
                _dataFeatures.append( _sample[0] )
                _dataLabels.append( _sample[1] )

        assert ( len( _dataFeatures ) == len( _dataLabels ) ), 'ERROR: features-labels len mismatch'

        self.m_sessionDataX = _dataFeatures
        self.m_sessionDataY = _dataLabels

        # make schedule for sessions
        _opts_nbinsColors = [ 32, 64, 128, 255 ]
        _opts_nbinsNormals = [ 50, 100, 150, 250 ]
        _opts_kernel = [ 'linear' ]
        _opts_C = [ 1.0 ]
        _opts_gamma = [ 1.0 ]
        _opts_dataPercent = [ 0.1, 0.25, 0.5, 0.75, 1.0 ]

        for _opts in tqdm( list( itertools.product( *[ _opts_nbinsColors,
                                                       _opts_nbinsNormals,
                                                       _opts_kernel,
                                                       _opts_C,
                                                       _opts_gamma,
                                                       _opts_dataPercent ] ) ) ) :

            _sess = PSession()
            _sess.nbinsColors = _opts[0]
            _sess.nbinsNormals = _opts[1]
            _sess.kernel = _opts[2]
            _sess.C = _opts[3]
            _sess.gamma = _opts[4]
            _sess.dataPercent = _opts[5]
            _sess.trainSize = int( _opts[5] * len( _dataFeatures ) )

            self._makeTrainSession( _sess )
        
        raw_input( 'Press enter to continue...' )

    """
    Makes a training session from the given session parameters

    :param session : Session object
    """
    def _makeTrainSession( self, session ) :
        print( 'START SESSION: ', self._getSessionId( session ) + ' ****************' )
        # create the batch of data to use
        _size = int( math.floor( session.dataPercent * len( self.m_sessionDataX ) ) )
        session.trainX = np.array( self.m_sessionDataX[0:_size] )
        session.trainY = np.array( self.m_sessionDataY[0:_size] )
        session.trainX = self._generateSessionBatch( session.nbinsColors,
                                                     session.nbinsNormals,
                                                     session.trainX )
                                                     
        # print( 'shape_x_train: ', session.trainX.shape )
        # print( 'shape_y_train: ', session.trainY.shape )
        # print( 'X' )
        # print( session.trainX )
        # print( 'Y' )
        # print( session.trainY )
        # sys.exit( 0 )
        
        # create input scaler ( zero mean and unit variance normalization )
        session.inScaler = StandardScaler()
        session.inScaler.fit( session.trainX )
        # create output labels one-hot encoding
        session.outEncoder = LabelEncoder()
        session.trainY = session.outEncoder.fit_transform( session.trainY )
        # create the svm model
        session.model = svm.SVC( kernel = session.kernel, 
                                 C = session.C, 
                                 gamma = session.gamma )
        # 5-fold cross validation
        session.kf = cross_validation.KFold( len( session.trainX ),
                                             n_folds = 5,
                                             shuffle = True,
                                             random_state = 1 )
        session.scores = cross_validation.cross_val_score( cv = session.kf,
                                                           estimator = session.model,
                                                           X = session.trainX,
                                                           y = session.trainY,
                                                           scoring = 'accuracy' )
        session.predictions = cross_validation.cross_val_predict( cv = session.kf,
                                                                  estimator = session.model,
                                                                  X = session.trainX,
                                                                  y = session.trainY )
        # accuracy score
        session.accuracyScore = metrics.accuracy_score( session.trainY, session.predictions )
        # show results
        self._showSessionResults( session )
        print( '**************************************************************' )

    """
    Generate new samples from the given samples by ...
    reshaping using different histograms sizes
    """
    def _generateSessionBatch( self, nbinsColors, nbinsNormals, dataX ) :
        # We start with a huge bag of data with ...
        # DATASET_COLOR_HIST_BINS, DATASET_NORMAL_HIST_BINS ...
        # as base size of the histograms. From there, we modify the ...
        # size according to the required sizes
        _batchX = []
        for _x in dataX :
            # extract color and normals histogram
            _x_chist = _x[0:(3 * DATASET_COLOR_HIST_BINS)]
            _x_nhist = _x[(3 * DATASET_COLOR_HIST_BINS):]
            _x_chist_channels = np.array_split( _x_chist, 3 )
            _x_nhist_channels = np.array_split( _x_nhist, 3 )
            # convert each histogram channel to the desired sizes
            _sx_chist_channels = [ hist2hist( _x_chist_channels[i], nbinsColors )
                                   for i in range( len( _x_chist_channels ) ) ]
            _sx_nhist_channels = [ hist2hist( _x_nhist_channels[i], nbinsNormals )
                                   for i in range( len( _x_nhist_channels ) ) ]
            _sx_chist = np.concatenate( _sx_chist_channels ).astype( np.float64 )
            _sx_nhist = np.concatenate( _sx_nhist_channels ).astype( np.float64 )
            _batchX.append( np.concatenate( [ _sx_chist, _sx_nhist ] ) )

        return np.array( _batchX )

    def _showSessionResults( self, session ) :
        # print scores
        print( 'Results: ' + self._getSessionId( session ) )
        print( 'Scores: ' + str( session.scores ) )
        print( 'Accuracy: %0.2f (+/- %0.2f)' % ( session.scores.mean(), 2 * session.scores.std() ) )
        print( 'Accuracy score: ' + str( session.accuracyScore ) )
        # show confusion matrices
        self._showConfusionMatrices( session )

    def _getSessionId( self, session ) :
        _sessId = 'sess_'
        _sessId += 'kernel_' + session.kernel + '_'
        _sessId += 'nc_' + str( session.nbinsColors ) + '_'
        _sessId += 'nn_' + str( session.nbinsNormals ) + '_'
        _sessId += 'size_' + str( session.trainSize )
        return _sessId

    def _showConfusionMatrices( self, session ) :
        # compute confusion matrices
        _cMatrix = metrics.confusion_matrix( session.trainY, session.predictions )
        _cMatrixNormalized = _cMatrix.astype( 'float' ) / _cMatrix.sum( axis = 1 )[:, np.newaxis]
        # plot confusion matrices
        self._plotConfusionMatrix( _cMatrix, 
                                   session.outEncoder.classes_,
                                   'ConfusionMatrix - ' + self._getSessionId( session ) )
        self._plotConfusionMatrix( _cMatrixNormalized,
                                   session.outEncoder.classes_,
                                   'NormalizedConfusionMatrix - ' + self._getSessionId( session ) )

    def _plotConfusionMatrix( self, matrix, classes, title ) :

        plt.figure()
        plt.imshow( matrix, 
                    interpolation = 'nearest', 
                    cmap = plt.cm.Blues )
        plt.title( title )
        plt.colorbar()

        _tickMarks = np.arange( len( classes ) )

        plt.xticks( _tickMarks, classes, rotation = 45 )
        plt.yticks( _tickMarks, classes )

        _thresh = matrix.max() / 2.
        for i, j in itertools.product( range( matrix.shape[0] ), 
                                       range( matrix.shape[1] ) ):
            plt.text( j, i, '{0:.2f}'.format( matrix[i, j] ),
                      horizontalalignment = 'center',
                      color = 'white' if matrix[i, j] > _thresh else 'black' )

        plt.tight_layout()
        plt.ylabel( 'True label' )
        plt.xlabel( 'Predicted label' )
        plt.savefig( title + '.png' )

    def _showTrainingScheduleResults( self, sessions ) :
        
        pass

    """
    Shows a training example. Plots the histograms for a given sample

    :param x : feature vector of the sample, composed of [colorhistogram : normalhistogram]
    :param y : label of the given sample
    """
    def showSample( self, x, y, nbins_chist, nbins_nhist ) :
        # extract the histograms from the feature vector
        _x_chist = x[0:(3 * nbins_chist)]
        _x_nhist = x[(3 * nbins_chist):]
        # show histograms
        self._showHistograms( _x_chist, channels = 3, name = 'color' )
        self._showHistograms( _x_nhist, channels = 3, name = 'normal' )

    def _showHistograms( self, hist, channels = 3, name = 'hist' ) :
        # divide into the required number of channels
        _hchannels = np.array_split( channels )
        # plot each histogram channel
        plt.figure()
        for i in range( channels ) :
            plt.subplot( '13' + str( i + 1 ) )
            plt.hist( _hchannels[i], len( _hchannels[i] ), alpha = 0.75 )

    def saveModel( self, filename ) :
        # prepare object to save
        _classifier = { 'model' : self.m_clfModel, 
                        'classes' : self.m_clfClasses, 
                        'scaler' : self.m_clfScaler }
        # dump it into a saved object
        pickle.dump( _classifier, open( filename, 'wb' ) )

    def loadModel( self, filename ) :
        # load saved pickle object
        _classifier = pickle.load( open( filename, 'rb' ) )
        # retrieve the parts of the classifier into our object
        self.m_clfScaler = _classifier['scaler']
        self.m_clfModel = _classifier['model']
        self.m_clfLabelEncoder = _classifier['labelEncoder']

    def train( self, X, Y ) :

        pass

    def predict( self, x ) :

        pass


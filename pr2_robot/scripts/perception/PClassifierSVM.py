
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
        _dataSet = self._rearrangeDataset( _dataSet )
        self.m_sessionDataX, self.m_sessionDataY = self._splitFeaturesLabels( _dataSet )

        # make schedule for sessions
        _opts_nbinsColors = [ 32, 128, 255 ]
        _opts_nbinsNormals = [ 50, 150, 250 ]
        _opts_kernel = [ 'linear' ]
        _opts_C = [ 1.0 ]
        _opts_gamma = [ 1.0 ]
        _opts_dataPercent = [ 0.1, 0.2, 0.4, 0.8, 1.0 ]

        _sessions = []

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
            _sess.trainSize = int( _opts[5] * len( _dataSet ) )

            self._makeTrainSession( _sess )
            _sessions.append( _sess )
        
        self._showTrainingScheduleResults( _sessions )

        raw_input( 'Press enter to continue...' )

    """
    Rearranges a dataset assumed in format [<dataC1>,<dataC2>,...,<dataC8>] such that
    each sample appears in order [c1,c2,...,c8, c1,c2,...,c8, ...]

    :param dataSet : original dataset to rearrange
    """
    def _rearrangeDataset( self, dataSet ) :
        # split dataset into samples per class
        _dataSetPerClasses = np.array_split( dataSet, DATASET_NUM_CLASSES )
        _rdataSet = []
        # rearrange the dataset to have interlaced samples
        for i in range( len( _dataSetPerClasses[0] ) ) :
            for j in range( DATASET_NUM_CLASSES ) :
                _rdataSet.append( _dataSetPerClasses[j][i] )

        return _rdataSet

    """
    Splits dataset into features and labels ( X, Y )

    :param dataSet : input dataset with features and labels in it
    """
    def _splitFeaturesLabels( self, dataSet ) :
        # extract data into features and labels
        _dataFeatures = []
        _dataLabels = []
        for _sample in dataSet :
            if np.isnan( _sample[0] ).sum() < 1 :
                _dataFeatures.append( _sample[0] )
                _dataLabels.append( _sample[1] )

        assert ( len( _dataFeatures ) == len( _dataLabels ) ), 'ERROR: features-labels len mismatch'

        return _dataFeatures, _dataLabels

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
        # create input scaler ( zero mean and unit variance normalization )
        session.inScaler = StandardScaler().fit( session.trainX )
        session.trainX = session.inScaler.transform( session.trainX )
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
        # train classifier
        session.model.fit( X = session.trainX, y = session.trainY )
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
        # show results from worst to best in ascending order
        sessions.sort( key = lambda x: x.accuracyScore, reverse = True )
        for i in range( len( sessions ) ) :
            print( self._getSessionId( sessions[i] ), sessions[i].accuracyScore )
        # save session results
        _fhandle = open( 'results.txt', 'wb' )
        for i in range( len( sessions ) ) :
            _fhandle.write( self._getSessionId( sessions[i] ) + ' - ' + str( sessions[i].accuracyScore ) + '\n' )
        _fhandle.close()

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
        self._showHistograms( _x_chist, rmin = 0.0, rmax = 255.0, channels = 3, name = 'color' )
        self._showHistograms( _x_nhist, rmin = -1.0, rmax = 1.0, channels = 3, name = 'normal' )

    def _showHistograms( self, hist, rmin, rmax, channels = 3, name = 'hist' ) :
        # divide into the required number of channels
        _hchannels = np.array_split( channels )
        # plot each histogram channel
        for i in range( channels ) :
            plotHistogram( _hchannels[i], rmin, rmax )

    def saveModel( self, filename ) :
        # prepare object to save
        _classifier = { 'model' : self.m_clfModel, 
                        'classes' : self.m_clfClasses, 
                        'scaler' : self.m_clfScaler,
                        'labelEncoder' : self.m_clfLabelEncoder }
        # dump it into a saved object
        pickle.dump( _classifier, open( filename, 'wb' ) )

    def loadModel( self, filename ) :
        # load saved pickle object
        _classifier = pickle.load( open( filename, 'rb' ) )
        # retrieve the parts of the classifier into our object
        self.m_clfScaler = _classifier['scaler']
        self.m_clfModel = _classifier['model']
        self.m_clfLabelEncoder = _classifier['labelEncoder']

    """
    Trains the classifier on the dataset stored in the given datafile, ...
    whose features are assumed to be stored in the form [<dataC1>,<dataC2>,...,<dataC8>]

    : param datafile : filepath to the .sav file that holds the dataset
    """
    def train( self, datafile, modelname = 'model' ) :
        plt.ion()
        # get dataset from file
        _dataSet = pickle.load( open( datafile, 'rb' ) )
        _dataSet = self._rearrangeDataset( _dataSet )
        self.m_sessionDataX, self.m_sessionDataY = self._splitFeaturesLabels( _dataSet )
        # make session
        _sess = PSession()
        _sess.nbinsColors = 255
        _sess.nbinsNormals = 250
        _sess.kernel = 'linear'
        _sess.C = 1.0
        _sess.gamma = 1.0
        _sess.dataPercent = 1.0
        _sess.trainSize = int( 1.0 * len( _dataSet ) )
        # train using session
        self._makeTrainSession( _sess )
        # copy results back to model object
        self.m_clfLabelEncoder = _sess.outEncoder
        self.m_clfClasses = _sess.outEncoder.classes_
        self.m_clfScaler = _sess.inScaler
        self.m_clfModel = _sess.model
        # save the model for later usage
        self.saveModel( modelname + '.sav' )

    def predict( self, x ) :
        if self.m_clfModel is None :
            print( 'ERROR: There is no model trained to make predictions' )
            return

        _prediction = self.m_clfModel.predict( self.m_clfScaler.transform( x ) )
        _label = self.m_clfLabelEncoder.inverse_transform( _prediction )[0]

        return _label


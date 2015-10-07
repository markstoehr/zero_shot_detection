from collections import OrderedDict
import numpy
import theano
from theano import tensor

RANDOM_GENERATOR = numpy.random

def generate_zero_dataset(n_train_sequences,
                          n_test_sequences,
                          n_frames, n_channels):
    """Create a zero dataset. """
    train_sequences = numpy.zeros((n_train_sequences, n_channels, n_frames))
    test_sequences = numpy.zeros((n_test_sequences, n_channels, n_frames))
    return train_sequences, test_sequences

def normal_init(n_channels):
    """Model for a normal system"""
    params = OrderedDict()
    params["mu"] = theano.shared(RANDOM_GENERATOR.randn(n_channels), name="mu")
    params["sigma"] = theano.shared(numpy.exp(
        RANDOM_GENERATOR.rand(n_channels)/10), name="sigma")
    params["normal_loglike"] = 
    params[
    return params

def normal_likelihood_sequences(sequences, params):
    """Computes the likelihood over a sequence."""
    

def rnn_init(n_channels, n_nodes, n_components):
    """Create and initialize a simple recurrent network."""
    params = OrderedDict()
    # classifier
    params["U"] = numpy.random.randn(n_channels, n_nodes).astype(
        theano.config.floatX)
    params["b"] = numpy.zeros(n_nodes).astype(theano.config.floatX)
    params["Mu"] = numpy.zeros

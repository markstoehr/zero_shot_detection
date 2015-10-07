import numpy
import theano
from proposal_code import prototype

def test_normal_like():
    """Tests just running the normal likelihood."""
    n_train_sequences = 200
    n_test_sequences = 50
    n_frames = 20
    n_channels = 3
    train_sequences, test_seqences = prototype.generate_zero_dataset(
        n_train_sequences, n_test_sequences, n_frames, n_channels)
    params = prototype.normal_init(n_channels)
    prototype.normal_likelihood_sequences(train_sequences, params)

def test_simple_train():
    """Tests a simple training function."""
    n_train_sequences = 200
    n_test_sequences = 50
    n_frames = 20
    n_channels = 3
    train_sequences, test_seqences = prototype.generate_zero_dataset(
        n_train_sequences, n_test_sequences, n_frames, n_channels)

    n_nodes = 32
    rnn = prototype.rnn_init(n_channels, n_nodes)
    test_likelihood = prototype.rnn_test_likelihood(
        test_sequences, rnn)

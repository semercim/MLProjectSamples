import random
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers.core import Dropout, Dense
from keras.layers import Embedding, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from utils import read_dataset, transform, my_transformer
import joblib

# by Murat Semerci
# semercim@gmail.com

# Set the RNG to have consistent results in the runs
random.seed(42)  # meaning of life! From Hitchhiker's Guide to the Galaxy! :)


def create_deep_recurrent_network(vocab_size, embed_size, seq_len, activation='relu'):
    """
    This function creates a GRU-based RNN.
    The architectural parameters (such as number of hidden GRU units and layers) are fixed by some experience and
    assumptions.

    Parameters:
    :param vocab_size: The number of words in our token dictionary
    :param embed_size: The embedding size, the literature assumes that embedding works better than one-hot coding.
    :param seq_len: The sequence length
    :param activation: the activations function used. 'relu' is preferred since it looks out faster to train then 'tanh'

    Returns:
    :return (keras.Sequential model): The compiled model is returned.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=seq_len))
    model.add(GRU(64, activation=activation, recurrent_dropout=0.2, return_sequences=True, stateful=False))
    model.add(Dropout(0.2))
    model.add(GRU(32, activation=activation,  recurrent_dropout=0.2, return_sequences=True, stateful=False))
    model.add(Dropout(0.2))
    model.add(GRU(16, activation=activation, recurrent_dropout=0.2, stateful=False))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(most_freq_words=8192, max_len=128, batch_size=256, embedding_size=64, epochs=10):
    """
    This function trains a recurrent neural network (RNN).
    The RNN uses sequences in training.
    The strings are tokenized then converted into padded same length integer vectors.
    The trained RNN and the required transformation and labelencoder objects are also saved for later use in deployment

    Early stopping, validation set and dropout are used as regularization.

    Parameters:
    :param most_freq_words (int): the number of the most frequent tokens to be used in transformation (default 8192=2^13)
    :param max_len (int): the sequence length after conversion
    :param batch_size (int): the batch siye used in DNN training
    :param embedding_size (int): the size of the embedding in the RNN
    :param epochs (int): The number of epochs in training

    Returns:
    :return None: the trained RNN, transformer and labelencoders are saved.
    """
    # Read and create the datasets stratification
    X_train, y_train, X_val, y_val, X_test, y_test = read_dataset(replace_strs=[("'s", ""), (" an ", " "), (" a ", " "), (" the ", " ")])

    # Train the transformer and labelencoder
    labelencoder = LabelBinarizer().fit(np.unique(y_train))
    transformer = Tokenizer(num_words=most_freq_words)
    transformer.fit_on_texts(X_train)
    transformer_ = my_transformer(transformer=transformer, max_len=max_len)

    # Transform the training and validation dataset
    X_train, y_train = transform(transformer_, labelencoder, X_train, y_train)
    X_val, y_val = transform(transformer_, labelencoder, X_val, y_val)

    # Create callbacks for model checkpoint saves and early stopping
    my_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=2, min_delta=0.01),
        ModelCheckpoint(filepath='model.rnn.h5', save_best_only=True),
        TensorBoard(log_dir='log_rnn', write_images=True),
    ]

    # create and train the RNN
    model = create_deep_recurrent_network(most_freq_words, embed_size=embedding_size, seq_len=max_len)
    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                        verbose=1, callbacks=my_callbacks,
                        shuffle=True)

    # save the required objects for later use in deployment
    joblib.dump(transformer_, 'transformer.rnn.joblib')
    joblib.dump(labelencoder, 'labelencoder.rnn.joblib')

    # reload the saved object, a control to see whether they are safely saved
    model = load_model('model.rnn.h5')
    transformer_ = joblib.load('transformer.rnn.joblib')
    labelencoder = joblib.load('labelencoder.rnn.joblib')

    # the final evaluation on the test set
    X_test, y_test = transform(transformer_, labelencoder, X_test, y_test)
    test_score = model.evaluate(X_test, y_test)
    print("Test Acc = ", test_score)


if __name__ == '__main__':
    """
    The file trains a Recurrent Neural Network.
    For experimenting, please change the parameters in the train function.
    """

    # For the details of the parameters please refer to the function description above.
    train(most_freq_words=8192, max_len=128, batch_size=256, embedding_size=64, epochs=100)
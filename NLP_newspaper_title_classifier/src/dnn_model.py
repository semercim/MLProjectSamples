import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from utils import read_dataset, transform
import joblib

# by Murat Semerci
# semercim@gmail.com

random.seed(42)  # meaning of life! From Hitchhiker's Guide to the Galaxy! :)


def create_deep_network(vocab_size, activation='relu'):
    """
    This function creates a DNN.
    The architectural parameters (such as number of hidden units and layers) are fixed by some experience and
    assumptions.

    Parameters:
    :param vocab_size: The number of words in our token dictionary
    :param activation: the activations function used. 'relu' is preferred

    Returns:
    :return (keras.Sequential model): The compiled model is returned.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=vocab_size, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(max_df=0.6, batch_size=256, epochs=10):
    """
    This function trains a deep neural network (DNN).
    The DNN uses tfidfvectorized sparse vectors in training..
    The strings are tokenized then converted in tfidf weighted vectors.
    The trained DNN and the required transformation and labelencoder objects are also saved for later use in deployment

    Early stopping, validation set and dropout are used as regularization.

    Parameters:
    :param max_df (float): The upper threshold ratio to eliminate the token. The more common a token is,
                          the less likely it is to have discriminative information
    :param batch_size (int): the batch siye used in DNN training
    :param epochs (int): The number of epochs in training

    Returns:
    :return None: the trained DNN, transformer and labelencoders are saved.
    """
    # Read and create the datasets stratification
    X_train, y_train, X_val, y_val, X_test, y_test = read_dataset(replace_strs=[("'s", ""), (" an ", " "), (" a ", " "), (" the ", " ")])
    # Train the transformer and labelencoder
    transformer = TfidfVectorizer(max_df=max_df, stop_words='english', use_idf=True)
    transformer.fit(X_train)
    labelencoder = LabelBinarizer().fit(np.unique(y_train))

    # Transform the training and validation dataset
    X_train, y_train = transform(transformer, labelencoder, X_train, y_train)
    X_val, y_val = transform(transformer, labelencoder, X_val, y_val)

    # Create callbacks for model checkpoint saves and early stopping
    my_callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.01),
        ModelCheckpoint(filepath='model.dnn.h5', save_best_only=True),
        TensorBoard(log_dir='log_dnn', write_images=True),
    ]
    # Get the size of word dictionary
    terms = transformer.get_feature_names()
    # create and train the DNN
    model = create_deep_network(vocab_size=len(terms))
    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(X_val, y_val),
                        verbose=1, callbacks=my_callbacks,
                        shuffle=True)

    # save the required objects for later use in deployment
    joblib.dump(transformer, 'transformer.dnn.joblib')
    joblib.dump(labelencoder, 'labelencoder.dnn.joblib')

    # reload the saved object, a control to see whether they are safely saved
    model = load_model('model.dnn.h5')
    transformer = joblib.load('transformer.dnn.joblib')
    labelencoder = joblib.load('labelencoder.dnn.joblib')

    # the final evaluation on the test set
    X_test, y_test = transform(transformer, labelencoder, X_test, y_test)
    test_score = model.evaluate(X_test, y_test)
    print("Test Acc = ", test_score)


if __name__ == '__main__':
    """
    The file trains a Deep Neural Network.
    For experimenting, please change the parameters in the train function.
    """

    # For the details of the parameters please refer to the function description above.
    train(max_df=0.6, batch_size=256, epochs=100)
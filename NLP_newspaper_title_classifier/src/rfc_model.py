import random
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import read_dataset, transform

# by Murat Semerci
# semercim@gmail.com

# Set the RNG to have consistent results in the runs
random.seed(42)  # meaning of life! From Hitchhiker's Guide to the Galaxy! :)


def train(max_df=0.6, n_estimators=100, min_samples_split=100, min_samples_leaf=50):
    """
    This function trains a random forest classifier (RFC).
    The RFC uses tfidfvectorized sparse vectors in training.
    The strings are tokenized then converted in tfidf weighted vectors.
    The trained RFC and the required transformation and labelencoder objects are also saved for later use in deployment

    Parameters:
    :param max_df (float): The upper threshold ratio to eliminate the token. The more common a token is,
                          the less likely it is to have discriminative information
    :param n_estimators (int): The number of the trees to be used in the random forest
    :param min_samples_split (int): The mininum number of instances in a node to split, used as regularization
    :param min_samples_leaf (int): The mininum number of instances in a leaf, used as regularization

    Returns:
    :return None: the trained RFC, transformer and labelencoders are saved.
    """

    # Read and create the datasets stratification
    X_train, y_train, X_val, y_val, X_test, y_test = read_dataset(replace_strs=[("'s", ""), (" an ", " "), (" a ", " "), (" the ", " ")])
    # Train the transformer and labelencoder
    transformer = TfidfVectorizer(max_df=max_df, stop_words='english', use_idf=True)
    transformer.fit(X_train)
    labelencoder = LabelEncoder().fit(np.unique(y_train))

    # Transform the training and validation dataset
    X_train, y_train = transform(transformer, labelencoder, X_train, y_train)
    X_val, y_val = transform(transformer, labelencoder, X_val, y_val)

    # create the RFC
    rfc_model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample")

    # train the RFC
    rfc_model.fit(X_train, y_train)

    # save the required objects for later use in deploymnet
    joblib.dump(rfc_model, 'model.rfc.joblib')
    joblib.dump(transformer, 'transformer.rfc.joblib')
    joblib.dump(labelencoder, 'labelencoder.rfc.joblib')

    # reload the saved object, a control to see whether they are safely saved
    rfc_model = joblib.load('model.rfc.joblib')
    transformer = joblib.load('transformer.rfc.joblib')
    labelencoder = joblib.load('labelencoder.rfc.joblib')

    # print the training results
    y_predicted_tr = rfc_model.predict(X_train)
    print("Train Accuracy on RFC = ", np.sum(y_predicted_tr == y_train)/len(y_train))
    # print the validation results
    y_predicted_val = rfc_model.predict(X_val)
    print("Validation Accuracy on RFC = ", np.sum(y_predicted_val == y_val)/len(y_val))
    # the final evaluation on the test set
    X_test, y_test = transform(transformer, labelencoder, X_test, y_test)
    y_predicted = rfc_model.predict(X_test)
    print("Test Accuracy on RFC = ", np.sum(y_predicted == y_test)/len(y_test))


if __name__ == '__main__':
    """
    The file trains a Random Forest Classifier.
    For experimenting, please change the parameters in the train function.
    """

    # For the details of the parameters please refer to the function description above.
    train(max_df=0.6, n_estimators=100, min_samples_split=50, min_samples_leaf=25)

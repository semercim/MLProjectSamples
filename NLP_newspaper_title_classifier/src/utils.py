import string
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

# by Murat Semerci
# semercim@gmail.com

# Set the RNG to have consistent results in the runs
random.seed(42)  # meaning of life! From Hitchhiker's Guide to the Galaxy! :)


def transform_text(df, col_name='title', replace_strs=None):
    """
    This is a utility function that is commonly used by the model trainings.
    It takes the column of strings named col_name in df pandas dataframe
    It replaces some substrings and eliminates the punctuations

    Parameters:
    :param df (pandas.dataframe): the pandas data frame
    :param col_name (string):  the name of the string column
    :param replace_strs (list(tuple(str, str))): list of pairs of substrings to replace (original_substr, substitute_substr)

    Returns:
    :return (None): The df is col_name colums strings are processed
    """

    # lowercase the string
    df[col_name] = df[col_name].map(lambda x: x.lower())
    # replace the substrs. E.g. " 's" is replaced with "".
    for (substr2change, replacement) in replace_strs:
        df[col_name] = df[col_name].map(
            lambda x: x.replace(substr2change, replacement)
        )
    # remove the punctuations
    df[col_name] = df[col_name].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))


def read_dataset(data_file_path='../data/newsCorpora.csv', replace_strs=None):
    """
    This is a utility function that is commonly used by the model trainings.
    It reads the 'cvs' data file located in the data_file_path and takes the
    title and category columns. The title columns is prepared for further transformation

    Parameters:
    :param data_file_path (string): the location of the csv file
    :param replace_strs (list(tuple(str, str))): list of pairs of substrings to replace (original_substr, substitute_substr)

    Returns:
    :return tuple(6 x numpy.array): X_train, y_train, X_val, y_val, X_test, y_test numpy arrays for data and labels
    """

    # read csv file and extract title and category titles only
    data = pd.read_csv(filepath_or_buffer=data_file_path, delimiter='\t', header=None)
    data.columns = ['id', 'title', 'url', 'publisher', 'category', 'story', 'hostname', 'timestep']
    data = data[['title', 'category']]

    # lowercase the titles and eliminates some substrs and punctuations
    transform_text(data, col_name='title', replace_strs=replace_strs)

    # convert from pandas data frame to numpy arrays
    X, y = data['title'].values, data['category'].values

    # partition data as training, validation and test datasets with stratification
    # 60%, 20%, 20% of the original data is splitted as train, val and test, respectively
    X_pre_train, X_test, y_pre_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_pre_train, y_pre_train, test_size=0.25,
                                                      random_state=42, stratify=y_pre_train)

    # return the data sets
    return X_train, y_train, X_val, y_val, X_test, y_test


class my_transformer:
    """
    This is just a wrapper class to wrap the operations to convert tokenized strings
    into integer sequences, which are used in RNN for sequence classification.
    With this wrapper, the same transform functions are used in later steps
    """
    def __init__(self, transformer, max_len):
        self.transformer = transformer # keras tokenizer object
        self.max_len = max_len # integer, length of the sequences

    def transform(self, X):
        """
        A common transform interface

        Parameters:
        :param X (array of strings): the input to be transformed into sequence array

        Returns:
        :return (padded numpy array): the inputs are converted into an array, where each row is a padded seqence of integers
        """
        sequences = self.transformer.texts_to_sequences(X)
        X_seq = pad_sequences(sequences, maxlen=self.max_len, dtype='int32', value=0)
        return X_seq


def transform(transformer, labelencoder, X, y):
    """
    This is the common transformation and label encoding function.
    The transformation and label encoding depends on the transformer and labelencoder parameters.

    Parameters
    :param transformer: sklearn or my_transformer object to be used in transformation
    :param labelencoder: label encoder (sklearn LabelEncoder or LabelBinarizer)
    :param X (string array): strings to be transformed
    :param y (label array): label to be encoded (srings labels to numerical labels or one-hot labels)

    Returns:
    :return tuple(array, array): X_transformed, y_encoded
    """
    X_transformed = transformer.transform(X)
    y_encoded = labelencoder.transform(y)
    return X_transformed, y_encoded
#!flask/bin/python
from flask import Flask, jsonify, abort, make_response, request
import joblib
from keras.models import load_model
import tensorflow as tf

# by Murat Semerci
# semercim@gmail.com

app = Flask(__name__)


def get_label(y):
    """
    This is a helper function to convert the label chars into a more meaningful labels
    Parameter:
    :param y (char): char label
    Returns:
    :return string: the label is reshaped to full extend
    """
    if y == 'e':
        return 'entertainment'
    elif y == 'b':
        return 'business'
    elif y == 't':
        return 'technology'
    return 'health'


@app.route('/', methods=['GET'])
def get_urls():
    """
    This is the home page response.
    :return json object: A json object showing the services
    """
    return jsonify({'rfc': '/rfc',
                    'dnn': '/dnn',
                    'rnn': '/rnn'})


@app.errorhandler(404)
def not_found(error):
    """
    This is the error response.
    """
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/rfc', methods=['POST'])
def rfc_predict():
    """
    This is the response function that loads the trained random forest classifier (RFC).
    It expects a request json a with 'title' key in it.

    :return json object: returns a dict of prediction_char and prediction_label
                        prediction_char is the original label returned by the RFC
                        prediction_label is the human readable label
    """
    if not request.json or not 'title' in request.json:
        abort(400)
    model = joblib.load('model.rfc.joblib')
    transformer = joblib.load('transformer.rfc.joblib')
    labelencoder = joblib.load('labelencoder.rfc.joblib')
    title = request.json['title']
    X = transformer.transform(title)
    prediction = model.predict(X)
    predicted_label_char = labelencoder.inverse_transform(prediction)[0]
    predicted_label = get_label(predicted_label_char)
    return jsonify({'prediction_char': predicted_label_char,
                    'prediction_label': predicted_label})


@app.route('/rnn', methods=['POST'])
def rnn_predict():
    """
    This is the response function that loads the trained recurrent neural network (RNN).
    It expects a request json a with 'title' key in it.

    :return json object: returns a dict of prediction_char and prediction_label
                        prediction_char is the original label returned by the RNN
                        prediction_label is the human readable label
    """
    if not request.json or not 'title' in request.json:
        abort(400)
    model = load_model('model.rnn.h5')
    transformer = joblib.load('transformer.rnn.joblib')
    labelencoder = joblib.load('labelencoder.rnn.joblib')
    title = request.json['title']
    X = transformer.transform(title)
    prediction = model.predict(X)
    predicted_label_char = labelencoder.inverse_transform(prediction)[0]
    predicted_label = get_label(predicted_label_char)
    return jsonify({'prediction_char': predicted_label_char,
                    'prediction_label': predicted_label})


@app.route('/dnn', methods=['POST'])
def dnn_predict():
    """
    This is the response function that loads the trained deep neural network (DNN).
    It expects a request json a with 'title' key in it.

    :return json object: returns a dict of prediction_char and prediction_label
                        prediction_char is the original label returned by the DNN
                        prediction_label is the human readable label
    """
    if not request.json or not 'title' in request.json:
        abort(400)
    model = load_model('model.dnn.h5')
    transformer = joblib.load('transformer.dnn.joblib')
    labelencoder = joblib.load('labelencoder.dnn.joblib')
    title = request.json['title']
    X = transformer.transform(title)
    prediction = model.predict(X)
    predicted_label_char = labelencoder.inverse_transform(prediction)[0]
    predicted_label = get_label(predicted_label_char)
    return jsonify({'prediction_char': predicted_label_char,
                    'prediction_label': predicted_label})


if __name__ == '__main__':
    app.run(host='0.0.0.0')

import requests
import json

# by Murat Semerci
# semercim@gmail.com

# This demo script classifies a given title string in each of the three models,
# which are already deployed in a REST service

# The default port number
port_number = 5000

# The endpoint urls for the classification services
# The Random Forest Classifier
endpoint_rfc = 'http://localhost:' + str(port_number) + '/rfc'
# The Recurrent Network Classifier
endpoint_rnn = 'http://localhost:' + str(port_number) + '/rnn'
# The Deep Neural Network classifier
endpoint_dnn = 'http://localhost:' + str(port_number) + '/dnn'

# Common header section
headers = {'Content-Type': 'application/json'}

# Convert the title sring to a serializable list in a JSON object#
# Please change the title string to run a new classification query
title = ['Juan Pablo Galavis Hits Vegas Before Bachelor Finale']
input_json = json.dumps({"title": title})


# The RFC is queried and the returned prediction is printed
predictions_rfc = requests.post(endpoint_rfc, input_json, headers=headers).json()
print('RFC prediction:', predictions_rfc['prediction_char'] + ' - ' + predictions_rfc['prediction_label'])

# The RNN is queried and the returned prediction is printed
predictions_rnn = requests.post(endpoint_rnn, input_json, headers=headers).json()
print('RNN prediction:', predictions_rnn['prediction_char'] + ' - ' + predictions_rnn['prediction_label'])

# The DNN is queried and the returned prediction is printed
predictions_dnn = requests.post(endpoint_dnn, input_json, headers=headers).json()
print('DNN prediction:', predictions_dnn['prediction_char'] + ' - ' + predictions_dnn['prediction_label'])

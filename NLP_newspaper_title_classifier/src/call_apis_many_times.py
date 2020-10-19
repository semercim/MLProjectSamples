import requests
import json

# by Murat Semerci
# semercim@gmail.com

# This interactive demo script classifies a given title string in each of the three models,
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

# Print some info onto the screen
print("Which classifier do you want to use:")
print("1 - Deep Neural Network\n2 - Recurrent Neural Network\n3 - Random Forest Classifier\n4 - Exit")

# Select the classifier and store it in choice
choice = int(input('Please enter your choice:'))

while choice != 4:
    if choice == 1:
        # The DNN is queried and the returned prediction is printed
        title = input('Please enter the title:')
        input_json = json.dumps({"title": [title]})
        predictions_dnn = requests.post(endpoint_dnn, input_json, headers=headers).json()
        print('DNN prediction:', predictions_dnn['prediction_char'] + ' - ' + predictions_dnn['prediction_label'])
    elif choice == 2:
        # The RNN is queried and the returned prediction is printed
        title = input('Please enter the title:')
        input_json = json.dumps({"title": [title]})
        predictions_rnn = requests.post(endpoint_rnn, input_json, headers=headers).json()
        print('RNN prediction:', predictions_rnn['prediction_char'] + ' - ' + predictions_rnn['prediction_label'])
    elif choice == 3:
        # The RFC is queried and the returned prediction is printed
        title = input('Please enter the title:')
        input_json = json.dumps({"title": [title]})
        predictions_rfc = requests.post(endpoint_rfc, input_json, headers=headers).json()
        print('RFC prediction:', predictions_rfc['prediction_char'] + ' - ' + predictions_rfc['prediction_label'])
    elif choice == 4:
        # Exit the loop
        print('Exit selected. Exiting...')
        break
    else:
        print('Invalid Choice')
    choice = int(input('Please enter your choice:'))

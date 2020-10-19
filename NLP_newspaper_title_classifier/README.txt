############ Folder structure

-NLP:
    - README.md
    - data FOLDER: should contain `newsCorpora.csv` as the data file, you should put it here.  You can download it from https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
    - src FOLDER: contains all the python, yml etc. files

############ To train the models in own computer
!! IMPORTANT: To train the models, the source files assume that `newsCorpora.csv` exists in 'data' folder. Otherwise,
you need to specify the `data_file_path` in the corresponding function.

First, you need to setup your virtual environment with conda. The yaml file is already provided (win or mac).
    - Create the environment with conda `conda env create --file NLP_xxx.yml`
    - Activate the environment `conda activate NLP`

Then, you need to run the file:
    - Go to the corresponding classifier file. For the random forest classifier, rfc_model.py, for the recurrent neural
      network, rnn_model.py and for the deep neural network, dnn_model.
    - Change the parameter at the bottom of the file in the `train` function, in case you want to change the parameters
    - Finally, run the corresponding python file, E.g. `python dnn_model.py`
    - The corresponding classifier and the other required objects are saved into the folder.


############ To deploy the classifiers and to run it as a service
!! IMPORTANT: For the assignment purposes, it is assumed that the trained models are already stored in the 'src' folder
as 'h5' or 'joblib' files. Otherwise the service can crash.

There are two ways to start the rest api:
1) You can run it in a container with Docker:
    In the src folder:
        - Build image `docker build -t rest-api .`
        - Run container in detached mode and publish port 5000 `docker run -d -p 5000:5000 rest-api`
        - API should be accessible on port 5000 `localhost:5000/`
2) You can run it as a python process:
    In the root folder using the bash:
        - Create the environment with conda `conda env create --file NLP_xxx.yml`
        - Activate the environment `conda activate NLP`
        - In the src folder, run `python app.py`
        - API should be accessible on port 5000 `localhost:5000/`


########### To query the API with a script
!! To run a query first you have to deploy the service in the previous section just above.

There are two demo scripts to call the api, auto and interactive:
1) If you want to call the api and query all the classifiers at once without much intervention:
        - Open `call_apis_once.py` in an editor, change the `title` variable with the new string you want to query.
        - Then, in the console, run `python call_apis_once.py`
        - The new title will be queried with all the three classifier and the results will be printed to the screen
2) For interactive run:
        - In the console, run `python call_apis_many_times.py`
        - Then follow the instructions, select the classifier and enter the title to query.
        - Exit the program whenever you want.


Dr. Murat Semerci
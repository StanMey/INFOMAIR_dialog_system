# Dialog system (INFOMAIR) Group - G03
Dialog system created for the Methods in AI Research (INFOMAIR) course at the Utrecht University.

Teammembers:
- Nathalie Kirch
- Iris Folpmers
- Panagiotis Dallas
- Stan Meyberg

## Using the system
A `[env.yml]` file has been added to this repository in order to easily run the system.

### Install locally (using anaconda)
1. `conda env create -f env.yml`
2. `conda activate infomair_env`
3. `python -m spacy download en_core_web_sm`

### Run the dialog system
The following command should be run to interact and use the system:

```
python main.py
```

## Part 1a: Text classification
To classify the sentences into dialog acts, four different classifiers were developed (2 baselines and 2 based on machine learning).

- Baseline systems
    - The majority class baseline
    - The rule-based system based on keyword matching
- Machine learning systems
    - A Naive Bayes approach
    - A K-nearest neighbours (KNN) approach

In order to run the analysis over all the developed models, the following command can be run:
```
python ./intent_classification.py
```
This will train all the models on the train dataset and evaluate the models on the test dataset.

### Deliverables
- Python code that implements a majority class baseline and a keyword matching baseline
- Python code that implements two or more machine learning classifiers

## Part 1b: Dialog management
In order to implement the dialog management system, the following State Transition Diagram (STD) has been modelled:

<img src="State Transition Diagram Dialog System.svg" alt="The State Transition Diagram of the dialog system">

This diagram has been implemented with python into a working dialog management system using a state transition function. Furthermore a lookup function has been added to find suitable restaurant suggestions and a algorithm for identifying the preferences of the user based on their utterances.

### Deliverables
- The state transition diagram
- A working dialog system interface, implementing a state transition function.
- An algorithm identifying user preference statements in the sentences using pattern matching on variable keywords and value keywords on utterances
- A lookup function that retrieves suitable restaurant suggestions from the CSV database


## Part 1c: Reasoning and configurability

### Reasoning
In the dialog system a simple reasoning component has been implemented. This uses inference rules to determine additional restaurant properties based on initial properties. The user can choose for the following additional preferences:

- touristic
- untouristic
- assigned seats
- children
- no children
- romantic
- unromantic

### Configurability
In order to implement some degree of configurability in the dialog manager the following features can be changed in the `.env` file:

- `levenshtein_distance` (0 or higher)
    - Levenshtein edit distance for preference extraction (0 means a 'dynamic' edit distance is used)
- `formal` (True or False)
    - Use formal or informal phrases in system utterances
- `use_caps` (True or False)
    - OUTPUT IN ALL CAPS OR NOT
- `allow_restart` (True or False)
    - Allow dialog restarts
- `use_delay` (True or False)
    - Introduce a delay before showing system responses
- `use_tts` (True or False)
    - Use text-to-speech for system utterances

### Deliverables
- Implementation of implication rules
- Implementation of configurability for selected features
- Integration of the two deliverables above as part of the dialog management system from Part 1b

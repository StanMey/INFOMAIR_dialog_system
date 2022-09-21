from tkinter import dialog
import pandas as pd

from dialog_management import DialogManager
from intent_classification.ml_naive_bayes import NaiveBayesPredictor

# load in the train/test data
train_path = "./data/training_dialog.pkl"
train_df = pd.read_pickle(train_path)


if __name__ == "__main__":

    # setup the intent classifier
    intent_model = NaiveBayesPredictor()
    intent_model.train(train_df["utterance_content"], train_df["dialog_act"])

    # setup the DialogManager
    user_answer = "hello"
    dialog_manager = DialogManager("hello", intent_model)
    dialog_manager.next_state(user_answer)

    # initialize the base case for the loop
    while dialog_manager.get_current_state != "thankyou" or "bye":

        # let the system make its utterance
        print(dialog_manager.get_system_utterance())

        # get the user input and make it lowercase
        user_answer = str(input()).lower()
        
        # process the user input and set the next state
        dialog_manager.next_state(user_answer)

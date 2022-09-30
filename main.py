import pandas as pd

from decouple import config
from pathlib import Path

from utils import load_restaurants
from dialog_management import DialogManager
from intent_classification import NaiveBayesPredictor


# load in the training dialog data
train_path = Path("./data/training_dialog.pkl")
train_df = pd.read_pickle(train_path)

# load in the restaurant options
restaurants_path = Path("./data/restaurant_info.csv")
restaurants = load_restaurants(restaurants_path)

# run the main program
if __name__ == "__main__":

    # setup the intent classifier
    intent_model = NaiveBayesPredictor()
    intent_model.train(train_df["utterance_content"].to_list(), train_df["dialog_act"].to_list())

    # setup the DialogManager
    user_answer = None
    dialog_manager = DialogManager("1_welcome", intent_model, restaurants)
    dialog_manager.run_system_response(1)

    # while the current state is not exit, keep on looping
    while dialog_manager.get_current_state() != "exit":
        if dialog_manager.demand_answer:
            # get the user input and make it lowercase
            user_answer = str(input()).lower()

        # process the user input and set the next state
        dialog_manager.next_state(user_answer)
        if config('debug', cast=bool):
            print(f"state: {dialog_manager.get_current_state()}; user_prefs: {dialog_manager.user_preferences}")

import sys
import pandas as pd

from intent_classification.baselines import MostOccuringBaselinePredictor, RuleBasedBaselinePredictor
from intent_classification.ml_naive_bayes import NaiveBayesPredictor

# load in the train/test data
train_path = "./data/training_dialog.pkl"
test_path = "./data/training_dialog.pkl"

train_df = pd.read_pickle(train_path)
test_df = pd.read_pickle(test_path)

# build the menu
choices = {
    "1": ("max occurances", MostOccuringBaselinePredictor),
    "2": ("rule-based", RuleBasedBaselinePredictor),
    "3": ("naive Bayes", NaiveBayesPredictor),
    "exit": ("exit from system", sys.exit)
}


if __name__ == "__main__":
    print("Please select a method for the intent classification step:")
    
    # display the steps
    for option, choice in choices.items():
        print(f"option {option}: {choice[0]}")
    
    # wait for answer and handle answer accordingly
    print("Please select an option:")
    user_answer = input()

    # select the model
    model = choices.get(user_answer)[1]()

    # Train the model
    model.train(train_df.utterance_content.to_list(), train_df.dialog_act.to_list())

    # 
    print("What's on your mind?")
    while user_answer != "exit":
        user_answer = input()
        prediction = model.predict([user_answer])
        print(f"{prediction}\nInteresting, what else?")
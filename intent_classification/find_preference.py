import sys
import pandas as pd
import re
import Levenshtein
import numpy as np
# from math import nan, isnan

from ml_naive_bayes import NaiveBayesPredictor

df = pd.read_csv (r'../data/restaurant_info(1).csv')

train_path = "../data/training_dialog.pkl"
test_path = "../data/test_dialog.pkl"

train_df = pd.read_pickle(train_path)
test_df = pd.read_pickle(test_path)

max_levensthein = 4

def find_preference(pref, options):
    with_levenstein = False
    smallest_distance = np.inf
    best = 0
    current = False
    prediction = model.predict([user_answer])
    if prediction == "inform":
        for option in options:
            if re.search(r"^.*" + re.escape(option) + r".*", user_answer):
                pref = option
                current = True
                break
            else:
                for word in user_answer.split():
                    distance = Levenshtein.distance(word, option)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        best = option
        if best and not current:
            if smallest_distance < max_levensthein:
                pref = best
                with_levenstein = True
    return pref, with_levenstein





if __name__ == "__main__":

    # select the model
    model = NaiveBayesPredictor()

    # Train the model
    model.train(train_df.utterance_content.to_list(), train_df.dialog_act.to_list())

    print("Welcome! Please give your preferences for a restaurant selection, your preferences can consist of location, cuisine and pricerange")
    user_answer = input()

    area = 0
    food = 0
    pricerange = 0

    areas = ["east", "west" , "north" , "south" , "center"] # list(set(df['area']))
    foods = list(set(df['food']))
    priceranges = list(set(df['pricerange']))
    phones = list(set(df['phone']))
    addresses = list(set(df['addr']))
    postcodes = list(set(df['postcode']))
    print(areas[0:])

    while user_answer != "exit":

        area = find_preference(area, areas)
        food = find_preference(food, foods)
        pricerange = find_preference(pricerange, priceranges)
        print(food, area, pricerange)
        print("Interesting, what else?")
        user_answer = input()
        
        
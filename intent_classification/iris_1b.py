import sys
import pandas as pd
import re
import Levenshtein
import numpy as np

from ml_naive_bayes import NaiveBayesPredictor

# load in the train/test data
train_path = "../data/training_dialog.pkl"
test_path = "../data/test_dialog.pkl"

train_df = pd.read_pickle(train_path)
test_df = pd.read_pickle(test_path)

max_levensthein = 4

def find_preference(saved_area, saved_cuisine, saved_price_range):
    areas = ["east", "west" , "north" , "south" , "center"] # list(set(df['food'])) en importen
    cuisines = [] 
    price_ranges = []

    # while user_answer != "exit":
    smallest_distance = np.inf
    best_area = 0
    area_now = False
    prediction = model.predict([user_answer])
    if prediction == "inform":
        for area in areas:
            if re.search(r"^.*" + re.escape(area) + r".*", user_answer):
                saved_area = area
                area_now = True
                break
            else:
                distance = Levenshtein.distance(user_answer, area)
                if distance < smallest_distance:
                    smallest_distance = distance
                    best_area = area
        if best_area and not area_now:
            if smallest_distance < max_levensthein:
                saved_area = best_area
            # else: saved_area = 0
    return (saved_area, saved_cuisine, saved_price_range)



if __name__ == "__main__":

    # select the model
    model = NaiveBayesPredictor()

    # Train the model
    model.train(train_df.utterance_content.to_list(), train_df.dialog_act.to_list())

    print("Welcome! Please give your preferences for a restaurant selection, your preferences can consist of location, cuisine and pricerange")
    user_answer = input()

    areas = ["east", "west" , "north" , "south" , "center"]
    cuisines = []
    price_ranges = []
    # smallest_distance = np.inf
    # best_area = 0
    # area_save = 0
    # area_now = False
    area = 0
    cuisine = 0
    price = 0

    while user_answer != "exit":

        area, _, _ = find_preference(area, cuisine, price)
        print(area)
        print("Interesting, what else?")
        user_answer = input()

        # smallest_distance = np.inf
        # best_area = 0
        # area_save = 0
        # area_now = False
        # prediction = model.predict([user_answer])
        # if prediction == "inform":
        #     for area in areas:
        #         if re.search(r"^.*" + re.escape(area) + r".*", user_answer):
        #             area_save = area
        #             area_now = True
        #             break
        #         else:
        #             distance = Levenshtein.distance(user_answer, area)
        #             if distance < smallest_distance:
        #                 smallest_distance = distance
        #                 best_area = area
        #     if best_area and not area_now:
        #         if smallest_distance < 4:
        #             area_save = best_area
        #         else: area_save = 0
        # print(area_save)
        
        



    #     # if prediction == "inform":
    #     # match user_answer.split():
    #     #     case [_ , "east", _] | [_ , "west", _] | [_ , "north", _]  | [_ , "south", _]  | [_ , "center", _] as area:
    #     #             print(f"The area is{area}")
    #     #     case ["look", _] | ["get"]:
    #     #         print("lookielook")
    #     #     case ["go"]:
    #     #         print("goooo")
                    
    #     print(f"{prediction}\nInteresting, what else?")
    #     user_answer = input()
        
        
import sys
from dataclasses import dataclass
from decouple import config
from typing import List, Tuple, Union

from dataloaders import Restaurant
from preference_extraction import find_preference
from intent_classification.ml_naive_bayes import NaiveBayesPredictor


dialog_choices = {
    1 : "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area, price range or food type. How may I help you?",
    2 : "What part of town do you have in mind?",
    3 : "What kind of food would you like?",
    4 : "What do you want more?",
    5 : "<name> is a nice place in the <area> of town and the prices are <pricerange>",
    6 : "I'm sorry but there are no restaurants matching your description, would you like to change something?",
    7 : "Would you like some additional information about the place?",
    8 : "<name> is on <address>",
    9 : "The phone number of <name> is <phone>",
    10: "The postcode of <name> is <postcode>",
    11: "Anything else?",
    12: "Goodbye and have a nice day",
    13: "Sorry I couldn't understand that"
}


@dataclass
class UserPreference:
    """A dataclass for holding information about the preferences of the user.
    """
    area: str = None
    cuisine: str = None
    pricerange: str = None

    def has_unfilled_preferences(self) -> bool:
        """Check whether there are still unfilled preferences left.

        Returns:
            bool: whether the user has unfilled preferences.
        """
        return not (self.area and self.cuisine and self.pricerange)
    
    def fill_preferences(self) -> None:
        """Fills the remaining preferences so we can give a recommendation.
        """
        if not self.pricerange:
            self.pricerange = "any"

    def __str__(self) -> str:
        return f"(area: {self.area}; cuisine: {self.cuisine}; pricerange: {self.pricerange})"


class DialogManager:
    """The class for the DialogManager, the brains of the dialogue.
    """
    def __init__(self, initial_state: str, intent_model: NaiveBayesPredictor, restaurants: List[Restaurant]) -> None:
        self.state = initial_state
        self.intent_model = intent_model
        self.user_preferences = UserPreference()
        # 
        self.restaurants = restaurants
        self.remaining_restaurants = restaurants
        self.chosen_restaurant = None
        # 
        self.max_levenshtein = config('levenshtein_distance', cast=int)
        self.unique_areas = list(set([rest.area for rest in self.restaurants if rest.area != ""]))
        self.unique_cuisines = list(set([rest.cuisine for rest in self.restaurants if rest.cuisine != ""]))
        self.unique_priceranges = list(set([rest.pricerange for rest in self.restaurants if rest.pricerange != ""]))
        self.contact_information = ["phone", "address", "postcode"]
        # 
        self.demand_answer = True
    
    def get_current_state(self) -> str:
        """getter function for getting the state.

        Returns:
            str: the current state.
        """
        return self.state


    def next_state(self, user_utterance: str) -> None:
        """This function uses the current state and the input from the user to set the next state.

        Args:
            user_utterance (str): the input from the user.
        """
        # predict the state based on the user input
        #TODO fix intent classification
        dialog_act = self.intent_model.predict([user_utterance])[0]
        print(f"dialog act: {dialog_act}")

        if dialog_act in ("bye", "thankyou"):
            self.state = "exit"
        
        elif dialog_act == "restart":
            # immediately restart the dialogue from the beginning and erase all gathered information
            self.user_preferences = UserPreference()
            self.chosen_restaurant = None
            self.state == "1_welcome"
        
        elif dialog_act == "dontcare":
            # TODO handle dontcare
            ...


        elif self.state == "1_welcome":
            if dialog_act == "hello":
                self.run_system_response(1)
                self.demand_answer = True

            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = self.extract_preferences(user_utterance)
                self.update_user_preferences(preferences)

            if not self.user_preferences.area:
                # ask for the area preference
                self.state = "2_ask_area"
                self.run_system_response(2)
                self.demand_answer = True

            elif not self.user_preferences.cuisine:
                # ask for the area preference
                self.state = "3_ask_cuisine"
                self.run_system_response(3)
                self.demand_answer = True
            
            elif self.user_preferences.has_unfilled_preferences():
                # ask the user for additional preferences
                self.state = "4_ask_additional_prefs"
                self.run_system_response(4)
                self.demand_answer = True
            else:
                # got all information, now make a suggestion
                self.state = "5_make_suggestion"


        elif self.state == "2_ask_area":
            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = self.extract_preferences(user_utterance)
                self.update_user_preferences(preferences)
            
            self.state = "1_welcome"
            self.demand_answer = False


        elif self.state == "3_ask_cuisine":
            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = self.extract_preferences(user_utterance)
                self.update_user_preferences(preferences)
            
            self.state = "1_welcome"
            self.demand_answer = False


        elif self.state == "4_ask_additional_prefs":
            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = self.extract_preferences(user_utterance)
                self.update_user_preferences(preferences)
            
            # after we asked for additional preferences fill the remaining spots with 'any', since the user doesn't care about those
            self.user_preferences.fill_preferences()
            self.state = "1_welcome"
            self.demand_answer = False


        elif self.state == "5_make_suggestion":
            
            if dialog_act in ("ack", "affirm"):
                self.state = "6_give_information"
                self.run_system_response(7)
                self.demand_answer = True
            
            else:
                if dialog_act in ("inform", "reqalts"):
                    # extract and update the preferences of the user
                    preferences = self.extract_preferences(user_utterance)
                    self.update_user_preferences(preferences)

                # filter on the restaurants
                self.filter_restaurants()

                if not self.remaining_restaurants:
                    # no restaurants to choose from
                    self.run_system_response(6)
                    self.demand_answer = True

                else:
                    # there are one/some restaurants available
                    self.chosen_restaurant = self.remaining_restaurants.pop()
                    self.run_system_response(5)
                    self.demand_answer = True


        elif self.state == "6_give_information":
            if dialog_act == "request":
                # the user wants some information
                req = find_preference(self.contact_information, user_utterance, max_levensthein=self.max_levenshtein)
                if req == "phone":
                    # the user wants the phone number
                    self.run_system_response(9)
                elif req == "address":
                    # the user wants the address
                    self.run_system_response(8)
                elif req == "postcode":
                    # the user wants the postcode
                    self.run_system_response(10)
                else:
                    # couldn't understand
                    self.run_system_response(13)
                
                self.run_system_response(11)
                self.demand_answer = True

            else:
                # the user is satisfied and we can say goodbye
                self.run_system_response(12)
                self.state = "exit"


    def extract_preferences(self, user_utterance: str) -> Tuple[Union[None,str], Union[None,str], Union[None,str]]:
        """Extract the preferences of the user from their response.

        Args:
            user_utterance (str): The response of the user.

        Returns:
            Tuple[Union[None,str], Union[None,str], Union[None,str]]: The found response of the user for the area, cuisine and pricerange
        """
        area = find_preference(self.unique_areas, user_utterance, max_levenshtein=self.max_levenshtein)
        cuisine = find_preference(self.unique_cuisines, user_utterance, max_levenshtein=self.max_levenshtein)
        pricerange = find_preference(self.unique_priceranges, user_utterance, max_levenshtein=self.max_levenshtein)
        return area, cuisine, pricerange
    
    def update_user_preferences(self, preferences: Tuple[Union[None,str], Union[None,str], Union[None,str]]) -> None:
        """Updates the preferences of the user based on the preferences found.

        Args:
            preferences (Tuple[Union[None,str], Union[None,str], Union[None,str]]): The preferences found during the extraction.
        """
        area, cuisine, pricerange = preferences
        if area:
            self.user_preferences.area = area
        if cuisine:
            self.user_preferences.cuisine = cuisine
        if pricerange:
            self.user_preferences.pricerange = pricerange

    def filter_restaurants(self) -> None:
        """Filters the restaurants based on the preference of the user.
        """
        options = []
        for r in self.restaurants:
            if (self.user_preferences.area == r.area or self.user_preferences.area == "any") and \
                (self.user_preferences.cuisine == r.cuisine or self.user_preferences.cuisine == "any") and \
                (self.user_preferences.pricerange == r.pricerange or self.user_preferences.pricerange == "any"):
                # this restaurant is acceptable
                options.append(r)

        # update the restaurants the user can choose from
        self.remaining_restaurants = options

    def run_system_response(self, dialog_option: int) -> None:
        """Selects a sentence based on the input and, whenever needed, constructs the sentences by filling in the templates.

        Args:
            dialog_option (int): The specific dialog to run in the CLI
        """
        dialog_sentence = dialog_choices.get(dialog_option)

        if self.chosen_restaurant:
            restaurant_info = [
                ("<name>", self.chosen_restaurant.name), ("<area>", self.chosen_restaurant.area), ("<pricerange>", self.chosen_restaurant.pricerange),
                ("<address>", self.chosen_restaurant.address), ("<phone>", self.chosen_restaurant.phone), ("<postcode>", self.chosen_restaurant.postcode)]

            # run all the replacements
            for tag, info in restaurant_info:
                dialog_sentence = dialog_sentence.replace(tag, info)

        print(dialog_sentence)
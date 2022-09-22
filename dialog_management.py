import sys
from dataclasses import dataclass

from intent_classification.ml_naive_bayes import NaiveBayesPredictor


dialog_choices = {
    "1_welcome": "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?",
    "2_ask_area": "What part of town do you have in mind?",
    "3_ask_cuisine": "What kind of food would you like?",
    "4_ask_additional_prefs": "What do you want more?",
    "5_make_suggestion": "My suggestion",
    "6_give_address": "",
    "7_give_phone": ""
}


def extract_preferences(user_utterance):
    """filler function
    """
    return 1


@dataclass
class UserPreference:
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
    def __init__(self, initial_state: str, intent_model: NaiveBayesPredictor) -> None:
        self.state = initial_state
        self.intent_model = intent_model
        self.last_state = None
        self.demand_answer = True
        self.user_preferences = UserPreference()
    
    def get_current_state(self) -> str:
        """getter function for getting the state.

        Returns:
            str: the current state.
        """
        return self.state


    def next_state(self, user_utterance: str):
        """_summary_.

        Args:
            user_utterance (str): _description_
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
            self.state == "1_welcome"


        elif self.state == "1_welcome":
            if dialog_act == "hello":
                self.last_state = self.state
                self.run_system_response()
                self.demand_answer = True

            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = extract_preferences(user_utterance)
                self.update_user_preferences(preferences)

            if not self.user_preferences.area:
                # ask for the area preference
                self.state = "2_ask_area"
                self.run_system_response()
                self.demand_answer = True

            elif not self.user_preferences.cuisine:
                # ask for the area preference
                self.state = "3_ask_cuisine"
                self.run_system_response()
                self.demand_answer = True
            
            elif self.user_preferences.has_unfilled_preferences():
                # ask the user for additional preferences
                self.state = "4_ask_additional_prefs"
                self.run_system_response()
                self.demand_answer = True
            else:
                # got all information, now make a suggestion
                self.state = "5_make_suggestion"


        elif self.state == "2_ask_area":
            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = extract_preferences(user_utterance)
                self.update_user_preferences(preferences)
            
            self.state = "1_welcome"
            self.demand_answer = False


        elif self.state == "3_ask_cuisine":
            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = extract_preferences(user_utterance)
                self.update_user_preferences(preferences)
            
            self.state = "1_welcome"
            self.demand_answer = False


        elif self.state == "4_ask_additional_prefs":
            if dialog_act == "inform":
                # extract and update the preferences of the user
                preferences = extract_preferences(user_utterance)
                self.update_user_preferences(preferences)
            
            # after we asked for additional preferences fill the remaining spots with 'any'
            self.user_preferences.fill_preferences()
            self.state = "1_welcome"
            self.demand_answer = False


        elif self.state == "5_make_suggestion":
            sys.exit()
        
        elif self.state == "6_give_address":
            ...
        
        elif self.state == "7_give_phone":
            ...


    def update_user_preferences(self, preferences):
        # testing 
        if self.state == "2_ask_area":
            self.user_preferences.area = "west"
        if self.state == "3_ask_cuisine":
            self.user_preferences.cuisine = "indian"


    def run_system_response(self):
        print(dialog_choices.get(self.state))
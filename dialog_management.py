from dataclasses import dataclass

from intent_classification.ml_naive_bayes import NaiveBayesPredictor

def extract_preferences(user_utterance):
    """filler function
    """
    return 1


@dataclass
class UserPreference:
    area: str = None
    cuisine: str = None
    pricerange: str = None


class DialogManager:
    def __init__(self, initial_state: str, intent_model: NaiveBayesPredictor) -> None:
        self.state = initial_state
        self.intent_model = intent_model
        self.user_utterance = None
        self.system_utterance = None
        self.user_preferences = UserPreference()

    def get_system_utterance(self) -> str:
        return self.system_utterance

    def get_current_state(self) -> str:
        return self.state

    def next_state(self, user_input: str):
        # update the user_utterance variable
        self.user_utterance = user_input

        # predict the state based on the user input
        #TODO fix intent classification
        self.next_state = self.intent_model.predict([self.user_utterance])

        # handle the next predicted state
        if self.next_state == "hello":
            # handle the 'hello' state
            self.system_utterance = "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"

        elif self.next_state == "inform":
            # extract the preferences
            #TODO preference extraction
            preferences = extract_preferences(self.user_utterance)
            if not self.user_preferences.area:
                # ask after area preference of the user
                self.system_utterance = "What part of town do you have in mind?"

            elif not self.user_preferences.cuisine:
                # ask after cuisine preference of the user
                self.system_utterance = "What kind of food would you like?"


        elif self.state == "thankyou" or "bye":
            # 
            self.system_utterance = ""

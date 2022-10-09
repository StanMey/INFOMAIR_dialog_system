import csv
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


@dataclass
class Restaurant:
    """A dataclass which holds information about a single restaurant.
    """
    name: str
    pricerange: str
    area: str
    cuisine: str
    phone: str
    address: str
    postcode: str
    quality: str
    crowdedness: str
    length_of_stay: str

    def get_restaurant_properties(self) -> Tuple[str, str, str, str, str]:
        return self.pricerange, self.cuisine, self.quality, self.crowdedness, self.length_of_stay

    def __repr__(self):
        return f"(name='{self.name}', pricerange='{self.pricerange}', area='{self.area}', food='{self.cuisine}', phone='{self.phone}', address='{self.address}', postcode='{self.postcode}')"


def load_restaurants(file_path: Path) -> List[Restaurant]:
    """Load and process the csv containing information about restaurants.

    Args:
        file_path (Path): The path to the csv file.

    Returns:
        List[Restaurant]: A list with Restaurant classes
    """
    restaurants = []

    # open the csv file and load all restaurants
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # skip the first line
                line_count += 1
            else:
                restaurants.append(Restaurant(*row))
    return restaurants


def load_restaurant_train_test_split(file_path: Path, test_size: float=0.15, seed: int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function to load and process the dialog_acts.dat file.
    strips the dialog acts from the utterances and does a stratified train/test split

    Args:
        file_path (Path): The path to the dialog_acts.dat file
        test_size (float, optional): The size of the test dataset. Defaults to 0.15.
        seed (int, optional): The value of the random seed. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Returns the train and test datasets.
    """
    
    # Load the dataset
    with open(file_path) as f:
        # convert to lower and remove the \n
        data = list(map(lambda x: x.lower().replace("\n", ""), f.readlines()))
    
    # extract the dialog acts from the string
    data = [(x.split(" ")[0], " ".join(x.split(" ")[1:])) for x in data]

    # set it to a pandas dataframe
    header = ["dialog_act", "utterance_content"]
    df_data = pd.DataFrame(data, columns=header)

    # do the first split
    df_train, df_test = train_test_split(df_data,
                                    test_size=test_size,
                                    random_state=seed,
                                    stratify=df_data["dialog_act"])
    
    return df_train, df_test

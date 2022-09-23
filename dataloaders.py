import csv

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Restaurant:
    """A dataclass which holds information about a single restaurant.
    """
    name: str
    pricerange: str
    area: str
    food: str
    phone: str
    address: str
    postcode: str

    def __repr__(self):
        return f"(name='{self.name}', pricerange='{self.pricerange}', area='{self.area}', food='{self.food}', phone='{self.phone}', address='{self.address}', postcode='{self.postcode}')"


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
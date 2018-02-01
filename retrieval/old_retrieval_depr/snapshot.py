"""Create an object to represent the state of the associated parking lot at the
given time.

    Attributes:
        path
        num_spots
        spot_status
        lot_name
        weather
        time
    Methods:
        auto_setup
        set_weather
        set_time
        set_spot_status
        get_spot_image
"""

from xml.etree import ElementTree
import datetime as dt
from os import path

import cv2
import numpy as np


class Snapshot(object):

    def __init__(self, xml_file, lot_name, num_spots):
        """Set the path, num_spots, and lot_name attributes, initialize
        spot_status, and call auto_setup to set all other attributes."""
        self.path = xml_file
        self.num_spots = num_spots
        self.lot_name = lot_name
        # Spot x's occupancy status corresponds to the element at index x of
        # spot_status. Since spot numbering commences from 1, the 0th value of
        # spot_status is not to be considered of the data.
        self.spot_status = np.empty((num_spots,), dtype="uint8")
        self.auto_setup()

    def auto_setup(self):
        """Call setter functions for weather, time, and each element of
        spot_status."""
        self.set_weather()
        self.set_time()
        self.set_spot_status()

    # Setters

    def set_weather(self):
        """Use the related .xml's suffix for setting weather.

        The reorganized PKLot dataset uses filename suffixes instead of
        folders to categorize shapshots.
        """
        if self.path.endswith("c.xml"):
            self.weather = "cloudy"
        elif self.path.endswith("s.xml"):
            self.weather = "sunny"
        else:
            self.weather = "rainy"

    def set_time(self):
        """Set the time of the snapshot using the related .xml's filename.

        The time of the snapshot is set as a datetime object. Snapshots
        were taken at five minute intervals for each lot in the dataset.
        """
        ymd = tuple(map(int, self.path[-25:-15].split('-')))
        hmss = tuple(map(int, self.path[-14:-6].split('_')))
        self.time = dt.datetime(*ymd, *hmss)

    def set_spot_status(self):
        """Set the value of every element of spot_status to reflect occupancy
        at the snapshot's time.

        In the dataset, an XML file accompanies each snapshot. It gives
        details of the position of each lot, its identification number, and
        sometimes, its occupancy status. When the occupancy status is missing,
        the suffix of the file of the .jpeg of the spot at the time is used to
        determine it. These suffixes were added during reorganization of PKLot.
        Note the zeroth element of set_status does not represent any spot.
        """
        tree = ElementTree.parse(self.path)
        all_spots = tree.findall('space')
        for spot in all_spots:
            index = int(spot.attrib['id']) - 1
            try:
                self.spot_status[index] = int(spot.attrib['occupied'])
            except KeyError:
                path_empty = (self.path[:-6]
                              + "#"
                              + str(index).zfill(3)
                              + "_e.jpg")
                if path.isfile(path_empty):
                    self.spot_status[index] = 0
                else:
                    self.spot_status[index] = 1

    # Getters

    def get_spot_image(self, spot_num):
        """Return an np.ndarray representing the image of the spot of number
        spot_num in BGR."""
        spot_num -= 1
        if self.spot_status[spot_num]:
            suffix = "_o.jpg"
        else:
            suffix = "_e.jpg"
        img_path = (self.path[:-6]
                    + "#"
                    + str(spot_num).zfill(3)
                    + suffix)
        return cv2.imread(img_path)

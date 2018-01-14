"""The class definition of the objects used to represent the state of the 
    associated parking lot at the given time.
    
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
        """Sets the path, num_spots, and lot_name attributes. Also initialize
        spot_status, and calls auto_setup to set all other attributes.
        """
        self.path = xml_file
        self.num_spots = num_spots
        self.lot_name = lot_name
        # Spot x's occupancy status corresponds to the element at index x of 
        # spot_status. Since spot numbering commences from 1, the 0th value of 
        # spot_status is not to be considered of the data.
        self.spot_status = np.empty((num_spots + 1,), dtype="int8")
        self.auto_setup()
     
    def auto_setup(self):
        """Calls setter functions for weather, time, and each element of 
        spot_status
        """
        self.set_weather()
        self.set_time()
        self.set_spot_status()
    
    # Setters 

    def set_weather(self):
        """The reorganized PKLot dataset uses filename suffixes instead of 
        folders to categorize shapshots. These suffixes are used for setting
        weather.
        """
        if self.path.endswith("c.xml"):
            self.weather = "cloudy"
        elif self.path.endswith("s.xml"):
            self.weather = "sunny"
        else:
            self.weather = "rainy"
        
    def set_time(self):
        """The time of the snapshot is set as a datetime object. Snapshots 
        were taken at five minute intervals for each lot in the dataset.
        """
        ymd = tuple(map(int, self.path[-25:-15].split('-')))
        hmss = tuple(map(int, self.path[-14:-6].split('_')))
        self.time = dt.datetime(*ymd, *hmss)
    
    def set_spot_status(self):
        """In the dataset, an XML file accompanies each snapshot. It gives 
        details of the position of each lot, its identification number, and 
        sometimes, its occupancy status. When the occupancy status is missing,
        the suffix of the file of the .jpeg of the spot at the time is used to 
        determine it. These suffixes were added during reorganization of PKLot.
        """
        tree = ElementTree.parse(self.path)
        all_spots = tree.findall('space')
        for spot in all_spots:
            index = int(spot.attrib['id'])
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
        """Returns an np.ndarray representing the image of the spot in BGR."""
        if self.spot_status[spot_num]:
            suffix = "_o.jpg"
        else:
            suffix = "_e.jpg"     
        img_path = (self.path[:-6] 
                    + "#" 
                    + str(spot_num).zfill(3) 
                    + suffix)
        print(img_path)
        return cv2.imread(img_path)
        
    
    
        
    
"""Create a Lot object to represent a parking lot.

    Attributes:
        name
        num_spots
        path
        all_xml_addresses
        num_snapshots
        all_snapshots
    Methods:
        auto_setup
        set_all_addresses
        set_lot_snapshots
"""

import os

import numpy as np

import snapshot as ss


class Lot(object):

    # The working directory for this module is urops/retrieval/
    # However, this module is accessed from urops/ml_algorithms. Some minor
    # problems relating to sys.path are caused by this when importing these
    # modules. They will be ironed out with use.

    BASE_PATH = "../data/"

    def __init__(self, name, num_spots):
        """Set the name, num_spots, and path attributes. Also: call auto_setup
        to set all other attributes.
        """
        self.name = name
        self.num_spots = num_spots
        self.path = Lot.BASE_PATH + name + "/"
        self.auto_setup()

    def auto_setup(self):
        """Call setter functions for all_xml_addresses, num_snapshots, and
        all_snapshots.
        """
        self.set_all_addresses()
        self.set_lot_snapshots()

    def set_all_addresses(self):
        """Retrieve the .xml addresses of the lot's associated snapshots, and
        set the attributes of all_xml_addresses, num_snapshots, and
        initialize all_snapshots.
        """
        all_files = np.array(os.listdir(self.path))
        self.all_xml_addresses = [fname for fname in all_files
                                  if fname.endswith(".xml")]
        self.num_snapshots = len(self.all_xml_addresses)
        self.all_snapshots = np.empty((self.num_snapshots,), dtype="O")

    def set_lot_snapshots(self):
        """Set all_snapshots to contain Snapshot objects for each snapshot of
        the lot, through using the snapshots' .xml files.
        """
        for index, xml_file in enumerate(self.all_xml_addresses):
            xml_file = self.path + xml_file
            self.all_snapshots[index] = ss.Snapshot(
                xml_file,
                self.name,
                self.num_spots)

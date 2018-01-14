"""This temporary module simply creates a Lot object for each lot in the 
dataset. It then serializes these objects for future use in other modules.
Each .pickle file is just 2 MB! 
"""

import pickle

from lot import Lot

solum = Lot("solum", 100)
primis = Lot("primis", 28)
secondus = Lot("secondus", 40)

solum_out = open("../data/solum/solum.pickle", "wb")
primis_out = open("../data/primis/primis.pickle", "wb")
secondus_out = open("../data/secondus/secondus.pickle", "wb")

pickle.dump(solum, solum_out)
pickle.dump(primis, primis_out)
pickle.dump(secondus, secondus_out)

solum_out.close()
primis_out.close()
secondus_out.close()

# Testing get_spot_image -- will probably need to create a pathway to obtain
# this directly from the Lot object, given a time.
solum.all_snapshots[0].get_spot_image(1)
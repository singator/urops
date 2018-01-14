"""Create and serialize a Lot object for each lot in the dataset."""

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

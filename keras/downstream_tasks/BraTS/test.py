#!/usr/bin/env python
"""
File: test
Date: 5/3/18 
Author: Jon Deaton (jdeaton@stanford.edu)

This file tests the functionality of the BraTS dataset loader
"""

import DataSet
import unittest
import numpy as np

# If, for some reason, you wanted to test this on your machine you would
# need to set up the BraTS data-sets in some directory and set that path here
brats_root = "/home/ioanna/PycharmProjects/uva-thesis/data"

class BraTSTest(unittest.TestCase):

    def test_patient(self):
        brats = DataSet(brats_root=brats_root, year=2017)
        patient = brats.train.patient("Brats17_TCIA_167_1")

        self.assertIsInstance(patient.id, str)
        self.assertIsInstance(patient.age, float)
        self.assertIsInstance(patient.survival, int)
        self.assertIsInstance(patient.mri, np.ndarray)
        self.assertIsInstance(patient.seg, np.ndarray)


if __name__ == "__main__":
    unittest.main()
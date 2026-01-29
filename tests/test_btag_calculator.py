import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Ensure the python directory is in path
sys.path.append(os.path.join(os.getcwd(), 'python'))

from helpers.btagWeightCalculator import BTagWeightCalculator

class TestBTagWeightCalculator(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.patcher_cl = patch('correctionlib.CorrectionSet')
        self.mock_cl = self.patcher_cl.start()

        self.patcher_uproot = patch('uproot.open')
        self.mock_uproot = self.patcher_uproot.start()

        self.patcher_os_exists = patch('os.path.exists')
        self.mock_os_exists = self.patcher_os_exists.start()
        self.mock_os_exists.return_value = True # Pretend files exist

        # Mock correction object
        self.mock_correction = MagicMock()
        self.mock_cl.from_file.return_value = MagicMock()
        self.mock_cl.from_file.return_value.__getitem__.return_value = self.mock_correction

        # Mock efficiency histograms
        # uroot.open returns a context manager that returns a file object
        # file object has keys() and __getitem__
        self.mock_root_file = MagicMock()
        self.mock_uproot.return_value.__enter__.return_value = self.mock_root_file
        self.mock_root_file.keys.return_value = ["jets_ptEta_genB"]

        # Mock histogram .to_numpy() -> (values, edges)
        # 1x1 bin for simplicity: value=0.8, x_edges=[0, 1000], y_edges=[0, 5]
        self.mock_hist = MagicMock()
        self.mock_hist.to_numpy.return_value = (
            np.array([[0.8]]),
            (np.array([0.0, 1000.0]), np.array([0.0, 5.0]))
        )
        self.mock_root_file.__getitem__.return_value = self.mock_hist

    def tearDown(self):
        self.patcher_cl.stop()
        self.patcher_uproot.stop()
        self.patcher_os_exists.stop()

    def test_init_with_wps(self):
        calc = BTagWeightCalculator("dummy.json", "dummy_path", "2018", is_run3=False, wp_medium=0.5, wp_tight=0.8)
        self.assertEqual(calc.wp_medium, 0.5)
        self.assertEqual(calc.wp_tight, 0.8)

    def test_init_defaults(self):
        # 2018 defaults
        calc = BTagWeightCalculator("dummy.json", "dummy_path", "2018", is_run3=False)
        self.assertEqual(calc.wp_medium, 0.2783)
        self.assertEqual(calc.wp_tight, 0.7100)

    def test_clamping_negative_weights(self):
        calc = BTagWeightCalculator("dummy.json", "dummy_path", "2018", is_run3=False, wp_medium=0.5)

        # Scenario:
        # Jet is NOT tagged (disc=0.1 < 0.5)
        # Efficiency = 0.8 (from mock)
        # SF = 2.0 (mock below)
        # P_MC = 1 - eff = 0.2
        # P_Data = 1 - sf*eff = 1 - 1.6 = -0.6 -> Should clamp to 0.0

        self.mock_correction.evaluate.return_value = 2.0 # SF

        jets_pt = [100.0]
        jets_eta = [1.0]
        jets_flav = [5]
        jets_btag = [0.1]

        weight = calc.calc_wp_weight(jets_pt, jets_eta, jets_flav, jets_btag, wp="M")

        # Expect weight = 0.0 / 0.2 = 0.0
        self.assertEqual(weight, 0.0)

    def test_clamping_negative_weights_tagged(self):
        # Scenario:
        # Jet IS tagged
        # This one naturally doesn't subtract, so it's safer, but let's test generic flow
        pass

    def test_shape_weight_simple_sys(self):
        calc = BTagWeightCalculator("dummy.json", "dummy_path", "2018")

        jets_pt = [100.0]
        jets_eta = [1.0]
        jets_flav = [4] # c-jet
        jets_btag = [0.5]

        self.mock_correction.evaluate.return_value = 1.1

        # Call with a specific systematic
        w = calc.calc_shape_weight(jets_pt, jets_eta, jets_flav, jets_btag, sys="up_jes")

        # Verify evaluate was called with "up_jes" and flavor 4
        # evaluate(sys, flav, eta, pt, disc)
        self.mock_correction.evaluate.assert_called_with("up_jes", 4, 1.0, 100.0, 0.5)
        self.assertEqual(w, 1.1)

if __name__ == '__main__':
    unittest.main()

import unittest
from python.helpers.utils import get_bjet_sf

class TestUtils(unittest.TestCase):
    def test_get_bjet_sf(self):
        self.assertAlmostEqual(get_bjet_sf(100, 0.5), 0.9544575)
        self.assertAlmostEqual(get_bjet_sf(800, 0.5), 0.98458995)
        self.assertEqual(get_bjet_sf(100, 3.0), 1.0)

if __name__ == '__main__':
    unittest.main()

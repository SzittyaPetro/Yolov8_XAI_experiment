from unittest import TestCase

import numpy as np

from LIME import load_images_from_paths


class Test(TestCase):
    def test_load_images_from_paths(self):
        image_paths = ["./data/gtFine/images/test/berlin/berlin_000000_000019_leftImg8bit.png",
                       "./data/gtFine/images/test/berlin/berlin_000001_000019_leftImg8bit.png"]
        images = load_images_from_paths(image_paths)

        self.assertEqual(len(images), len(image_paths))
        for image in images:
            self.assertIsInstance(image, np.ndarray)

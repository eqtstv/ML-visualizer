import unittest

import examples.keras_simple as test


class TestKerasSimple(unittest.TestCase):
    def test_data(self):
        self.assertEqual(test.train_images.shape, (60000, 28, 28))
        self.assertEqual(test.test_images.shape, (10000, 28, 28))
        self.assertEqual(test.train_labels.shape, (60000,))
        self.assertEqual(test.test_labels.shape, (10000,))

    def test_get_model(self):
        result = test.get_model()

        self.assertEqual(len(result.layers), 3)

    def test_imports(self):
        self.assertEqual(
            str(test.BatchTracker), "<class 'mlvisualizer.callback.BatchTracker'>"
        )
        self.assertEqual(
            str(test.EpochTracker), "<class 'mlvisualizer.callback.EpochTracker'>"
        )

import unittest
from PIL import Image
from object_detection import compare_predictions, render_detections


def item(label, score, box):
    return {'label': label, 'score': score, 'box': box}


class DetectionComparisonTests(unittest.TestCase):
    def test_match_by_class_and_overlap(self):
        before = {'detections': [item('cat', .9, [0, 0, 10, 10]), item('dog', .8, [20, 20, 30, 30])]}
        after = {'detections': [item('cat', .6, [1, 1, 11, 11]), item('bird', .7, [20, 20, 30, 30])]}
        result = compare_predictions(before, after)
        self.assertEqual((result['matched_count'], result['dropped_count'], result['added_count']), (1, 1, 1))
        self.assertAlmostEqual(result['severity'], .65)
        self.assertTrue(result['candidate'])

    def test_empty_and_identical_predictions(self):
        empty = {'detections': []}
        cat = {'detections': [item('cat', .8, [0, 0, 10, 10])]}
        self.assertFalse(compare_predictions(empty, empty)['candidate'])
        self.assertTrue(compare_predictions(empty, cat)['candidate'])
        self.assertEqual(compare_predictions(cat, cat)['severity'], 0)
        self.assertFalse(compare_predictions(cat, cat)['candidate'])

    def test_render_does_not_change_original_pixels(self):
        image = Image.new('RGB', (40, 40), 'white')
        rendered = render_detections(image, {'detections': [item('cat', .9, [5, 5, 25, 25])]})
        self.assertEqual(image.getpixel((5, 5)), (255, 255, 255))
        self.assertNotEqual(rendered.getpixel((5, 5)), (255, 255, 255))

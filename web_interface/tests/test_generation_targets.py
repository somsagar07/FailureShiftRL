"""Desired-image preprocessing preserves captioned objects and reports framing."""
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageOps


class GenerationTargetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('generation_targets'),
                             'Desired images need scene-preserving preprocessing')
        self.targets = importlib.import_module('generation_targets')
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)

    def save(self, image, name='target.png', **kwargs):
        path = self.directory / name
        image.save(path, **kwargs)
        return path

    def scene(self):
        image = Image.new('RGB', (800, 400), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 100, 79, 299), fill='red')
        draw.rectangle((720, 100, 799, 299), fill='blue')
        return image

    def test_default_preserves_both_captioned_objects_at_opposite_scene_edges(self):
        result, info = self.targets.prepare_target_image(self.save(self.scene()), 256)
        self.assertEqual(result.size, (360, 184))
        self.assertEqual(result.getpixel((8, 92)), (255, 0, 0))
        self.assertEqual(result.getpixel((351, 92)), (0, 0, 255))
        self.assertEqual(info['source_size'], [800, 400])
        self.assertEqual(info['oriented_size'], [800, 400])
        self.assertEqual(info['processed_size'], [360, 184])
        self.assertEqual(info['framing'], 'preserve')
        self.assertFalse(info['cropped'])
        self.assertIsNone(info['crop_box'])

    def test_explicit_center_crop_matches_previous_training_pixels_and_reports_loss(self):
        scene = self.scene()
        result, info = self.targets.prepare_target_image(
            self.save(scene), 256, framing='center_crop')
        expected = ImageOps.fit(scene, (256, 256), method=Image.Resampling.LANCZOS)
        self.assertEqual(result.tobytes(), expected.tobytes())
        self.assertEqual(info['crop_box'], [200.0, 0.0, 600.0, 400.0])
        self.assertTrue(info['cropped'])
        self.assertEqual(info['processed_size'], [256, 256])

    def test_exif_orientation_applies_before_sizing_and_records_stored_dimensions(self):
        image = Image.new('RGB', (80, 40), 'red')
        exif = image.getexif()
        exif[274] = 6
        path = self.save(image, 'portrait.jpg', exif=exif)
        result, info = self.targets.prepare_target_image(path, 128)
        self.assertEqual(result.size, (88, 184))
        self.assertEqual(info['source_size'], [80, 40])
        self.assertEqual(info['oriented_size'], [40, 80])
        self.assertTrue(info['exif_transposed'])
        self.assertNotIn(274, result.getexif())

    def test_small_images_are_upscaled_and_converted_to_detached_rgb(self):
        path = self.save(Image.new('L', (16, 16), 127))
        result, _ = self.targets.prepare_target_image(path, 64)
        path.unlink()
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.getpixel((0, 0)), (127, 127, 127))

    def test_dimensions_are_deterministic_quantized_and_bound_memory(self):
        for width, height, size, expected in (
                (800, 400, 256, (360, 184)),
                (400, 800, 256, (184, 360)),
                (3000, 1000, 1024, (1024, 344)),
                (32000, 1000, 512, (1024, 32)),
                (1000, 32000, 512, (32, 1024)),
                (17, 17, 32, (32, 32)),
                (200, 300, 512, (416, 624))):
            with self.subTest(width=width, height=height, size=size):
                actual = self.targets.target_dimensions(width, height, size)
                self.assertEqual(actual, expected)
                for dimension in actual:
                    self.assertEqual(dimension % 8, 0)
                    self.assertGreaterEqual(dimension, 32)
                    self.assertLessEqual(dimension, 1024)
                self.assertLessEqual(actual[0] * actual[1], (size + 8) ** 2)

    def test_extreme_aspect_ratios_fail_instead_of_stretching_or_cropping(self):
        for dimensions in ((10000, 100), (100, 10000)):
            with self.subTest(dimensions=dimensions):
                path = self.save(Image.new('RGB', dimensions))
                with self.assertRaisesRegex(ValueError, 'aspect ratio'):
                    self.targets.prepare_target_image(path, 512)

    def test_too_small_pixel_budget_fails_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, 'generation_size'):
            self.targets.target_dimensions(800, 400, 32)

    def test_invalid_framing_and_size_are_rejected(self):
        for framing in ('', None, True, 'fit', 'PRESERVE'):
            with self.subTest(framing=framing), self.assertRaisesRegex(ValueError, 'framing'):
                self.targets.target_dimensions(64, 64, 64, framing)
        for size in (None, True, 64.0, '64', 0, 24, 33, 1032):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, 'generation_size'):
                self.targets.target_dimensions(64, 64, size)
        for dimensions in ((0, 64), (64, -1), (True, 64), (64.0, 64)):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, 'dimensions'):
                self.targets.target_dimensions(*dimensions, 64)

    def test_unreadable_and_corrupt_images_report_validation_errors(self):
        corrupt = self.directory / 'corrupt.png'
        corrupt.write_bytes(b'not an image')
        truncated = self.save(Image.new('RGB', (64, 64)))
        truncated.write_bytes(truncated.read_bytes()[:55])
        for path in (self.directory / 'missing.png', self.directory, corrupt, truncated):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'image'):
                self.targets.prepare_target_image(path, 64)

    def test_large_decoded_image_is_rejected_before_any_pixel_allocation(self):
        # A PPM header advertises dimensions without allocating its claimed pixels.
        path = self.directory / 'oversized.ppm'
        path.write_bytes(b'P6\n5001 4000\n255\n')
        with patch.object(Image.Image, 'load', side_effect=AssertionError('decoded unsafe image')):
            with self.assertRaisesRegex(ValueError, '20 megapixels'):
                self.targets.prepare_target_image(path, 512)

    def test_decompression_bomb_is_a_normal_validation_error(self):
        path = self.directory / 'bomb.ppm'
        path.write_bytes(b'P6\n100000 100000\n255\n')
        with self.assertRaisesRegex(ValueError, 'image|megapixels'):
            self.targets.prepare_target_image(path, 512)

    def test_center_crop_square_reports_no_content_loss(self):
        result, info = self.targets.prepare_target_image(
            self.save(Image.new('RGBA', (64, 64), (255, 0, 0, 0))),
            32, framing='center_crop')
        self.assertEqual(result.mode, 'RGB')
        self.assertFalse(info['cropped'])
        self.assertEqual(info['crop_box'], [0.0, 0.0, 64.0, 64.0])


if __name__ == '__main__':
    unittest.main()

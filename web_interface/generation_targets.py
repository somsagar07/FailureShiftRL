"""Pillow-only preprocessing for reviewed diffusion training targets.

Preserve framing keeps the entire EXIF-oriented scene. The requested size is
the square root of the desired pixel budget, rather than a mandatory square
crop. Each output dimension is rounded to the nearest multiple of eight (ties
round up). The aspect ratio is therefore approximate and rounding can stretch
the scene slightly; its area may exceed ``size**2`` by that rounding alone.
"""
import math
from pathlib import Path
import warnings

from PIL import Image, ImageOps


MAX_IMAGE_PIXELS = 20_000_000
MIN_SIDE = 32
MAX_SIDE = 1024


def target_dimensions(width, height, size, framing='preserve'):
    """Return deterministic ``(width, height)`` compatible with a diffusion VAE.

    Inputs must be positive integer dimensions and an integer size in 32..1024
    divisible by eight. Preserve framing rejects aspect ratios or budgets that
    cannot fit both side bounds at a common scale. Independent rounding to
    multiples of eight approximates the aspect ratio and can stretch the scene.
    ``center_crop`` explicitly requests the legacy square target instead.
    """
    if framing not in ('preserve', 'center_crop'):
        raise ValueError('Target framing must be preserve or center_crop.')
    if type(size) is not int or not MIN_SIDE <= size <= MAX_SIDE or size % 8:
        raise ValueError('generation_size must be an integer multiple of 8 from 32 to 1024.')
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError('Image dimensions must be positive integers.')
    if framing == 'center_crop':
        return size, size

    # A ratio above 32 cannot fit [32, 1024] at any common scale.
    ratio = max(width, height) / min(width, height)
    if ratio > MAX_SIDE / MIN_SIDE:
        raise ValueError('Desired image aspect ratio cannot fit dimensions 32..1024; '
                         'use a less extreme image or explicitly choose center_crop framing.')
    scale = min(size / math.sqrt(width * height), MAX_SIDE / max(width, height))
    if min(width, height) * scale < MIN_SIDE - 1e-9:
        raise ValueError('Desired image aspect ratio needs a larger generation_size to '
                         'preserve the full scene with a minimum side of 32; '
                         'increase generation_size or explicitly choose center_crop framing.')
    return tuple(max(MIN_SIDE, min(MAX_SIDE, int(dimension * scale / 8 + .5) * 8))
                 for dimension in (width, height))


def _check_pixel_limit(image):
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError('Desired training images must be at most 20 megapixels.')


def prepare_target_image(path, size, framing='preserve'):
    """Validate, orient, and resize a target; return a detached RGB image and info.

    Metadata is JSON-safe. ``source_size`` is stored width/height;
    ``oriented_size`` applies EXIF orientation; ``processed_size`` describes the
    returned image. ``crop_box`` is None for preserve, or four coordinates in
    the oriented original image for center_crop. No padding is added.
    Preserve uses the full scene with approximate aspect ratio after independent
    dimension rounding to multiples of eight; this can slightly stretch it. The
    caller owns the returned image. Oversized image headers are rejected before
    verify, EXIF transposition, conversion, or pixel decoding allocates memory.
    """
    # Reject invalid settings even when the supplied path is unreadable.
    target_dimensions(1, 1, size, framing)
    try:
        image_path = Path(path)
        if not image_path.is_file():
            raise ValueError('Desired training image must be an existing readable image file.')
    except (TypeError, OSError) as exc:
        raise ValueError('Desired training image must be an existing readable image file.') from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(image_path) as source:
                _check_pixel_limit(source)
                source.verify()
            with Image.open(image_path) as source:
                _check_pixel_limit(source)
                source_size = source.size
                orientation = source.getexif().get(274, 1)
                image = ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('Desired training image must be a valid readable image '
                         'of at most 20 megapixels.') from exc

    oriented_size = image.size
    output_size = target_dimensions(*oriented_size, size, framing)
    crop_box = None
    cropped = False
    if framing == 'center_crop':
        width, height = oriented_size
        side = min(width, height)
        left, top = (width - side) / 2, (height - side) / 2
        crop_box = [left, top, left + side, top + side]
        cropped = width != height
        processed = ImageOps.fit(image, output_size, method=Image.Resampling.LANCZOS)
    else:
        processed = image.resize(output_size, resample=Image.Resampling.LANCZOS)
    image.close()
    return processed, {
        'framing': framing,
        'source_size': list(source_size),
        'oriented_size': list(oriented_size),
        'processed_size': list(output_size),
        'crop_box': crop_box,
        'cropped': cropped,
        'exif_transposed': orientation in (2, 3, 4, 5, 6, 7, 8),
    }

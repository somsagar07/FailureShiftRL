"""Fixed optional choices and review guidance, separate from enabled defaults."""
import copy


DEFAULT_IMAGE_SPACE = dict(rotation=[-10, -5, 0, 5, 10], darkness=[.1, .2, .3, .4, .5],
                           saturation=[.5, .55, .6, .65, .7])
GENERATION_STYLES = ['photograph', 'watercolor', 'pencil drawing', 'digital illustration', '3D render']
DEFAULT_GENERATION_DIMENSIONS = dict(style=GENERATION_STYLES,
    lighting=['even', 'soft', 'directional', 'high-key', 'low-key'],
    detail=['minimal', 'simple', 'moderate', 'detailed', 'intricate'])
GENERATION_DIMENSIONS = {
    **DEFAULT_GENERATION_DIMENSIONS,
    'composition': ['centered', 'balanced', 'rule of thirds', 'symmetrical'],
    'viewpoint': ['eye-level', 'high-angle', 'low-angle', 'overhead'],
    'framing': ['wide', 'medium', 'close-up'],
    'color_palette': ['natural', 'muted', 'vibrant', 'monochrome'],
    'focus': ['deep', 'shallow', 'soft', 'sharp'],
    'texture': ['smooth', 'matte', 'grainy', 'rough'],
}
DEFAULT_TEXT_ACTIONS = [
    'drop_stop_words', 'remove_punctuation', 'to_lowercase', 'negate_sentence',
    'delete_random_character', 'insert_random_character', 'change_letter_case',
    'apply_misspellings', 'repeat_random_character', 'replace_with_adjacent_character',
    'swap_random_character', 'delete_random_word', 'perturb_word_order', 'repeat_random_word',
]
TEXT_ACTIONS = DEFAULT_TEXT_ACTIONS + [
    'to_uppercase', 'to_titlecase', 'normalize_whitespace', 'expand_contractions',
    'contract_phrases', 'unicode_quotes', 'ascii_quotes', 'add_line_breaks',
    'add_sentence_numbers', 'bullet_sentences', 'reverse_sentences', 'duplicate_sentence',
    'truncate_last_sentence', 'replace_spaces_with_nbsp',
]

# Bounds and identity values also apply to direct Python consumers. The HTTP
# editor accepts the fixed suggested values below, as it did for the defaults.
IMAGE_DIMENSIONS = {
    'rotation': (-180, 180, 0), 'darkness': (0, 1, 0), 'saturation': (0, 3, 1),
    'brightness': (0, 3, 1), 'contrast': (0, 3, 1), 'sharpness': (0, 4, 1),
    'gaussian_blur': (0, 8, 0), 'gaussian_noise': (0, .2, 0),
    'jpeg_quality': (5, 100, 100), 'pixelation': (1, 16, 1),
    'hue_shift': (-.5, .5, 0), 'grayscale': (0, 1, 0), 'invert': (0, 1, 0),
    'posterize_bits': (1, 8, 8), 'solarize_threshold': (0, 256, 256),
    'translate_x': (-.3, .3, 0), 'translate_y': (-.3, .3, 0),
    'zoom': (.5, 1.5, 1), 'horizontal_flip': (0, 1, 0),
}
INTEGER_IMAGE_DIMENSIONS = {'jpeg_quality', 'pixelation', 'posterize_bits', 'solarize_threshold', 'horizontal_flip'}
IMAGE_CATALOG = {
    **DEFAULT_IMAGE_SPACE,
    'brightness': [.5, .75, 1, 1.25, 1.5],
    'contrast': [.5, .75, 1, 1.25, 1.5],
    'sharpness': [0, .5, 1, 2, 3],
    'gaussian_blur': [0, .5, 1, 2, 4],
    'gaussian_noise': [0, .02, .05, .1, .2],
    'jpeg_quality': [100, 75, 50, 25, 10],
    'pixelation': [1, 2, 4, 8, 16],
    'hue_shift': [-.2, -.1, 0, .1, .2],
    'grayscale': [0, .25, .5, .75, 1],
    'invert': [0, .25, .5, .75, 1],
    'posterize_bits': [8, 6, 4, 3, 2],
    'solarize_threshold': [256, 224, 192, 128, 64],
    'translate_x': [-.2, -.1, 0, .1, .2],
    'translate_y': [-.2, -.1, 0, .1, .2],
    'zoom': [.75, .9, 1, 1.1, 1.25],
    'horizontal_flip': [0, 1],
}


def _row(label, description, risk='low', reason=''):
    return dict(label=label, description=description, semantic_risk=risk, risk_reason=reason)


_IMAGE_METADATA = {
    'rotation': _row('Rotation', 'Rotate by degrees, keeping the image canvas.'),
    'darkness': _row('Darkness', 'Reduce brightness by this fraction.'),
    'saturation': _row('Saturation', 'Multiply color saturation.'),
    'brightness': _row('Brightness', 'Multiply brightness.', 'visibility', 'Brightness changes can hide or clip visual evidence.'),
    'contrast': _row('Contrast', 'Multiply contrast around the image mean.', 'visibility', 'Contrast changes can remove visual evidence.'),
    'sharpness': _row('Sharpness', 'Smooth or emphasize edges.', 'visibility', 'Smoothing or edge artifacts can change visible details.'),
    'gaussian_blur': _row('Gaussian blur', 'Blur with the selected pixel radius.', 'visibility', 'Blur can make objects, text, and details unreadable.'),
    'gaussian_noise': _row('Gaussian noise', 'Add fixed-seed noise; values are a fraction of the pixel range.', 'visibility', 'Noise can hide small objects and details.'),
    'jpeg_quality': _row('JPEG quality', 'Compress and decode once; 100 preserves source pixels.', 'visibility', 'Compression artifacts can obscure details.'),
    'pixelation': _row('Pixelation', 'Downsample by this factor and enlarge with nearest-neighbor sampling.', 'visibility', 'Pixelation removes fine visual evidence.'),
    'hue_shift': _row('Hue shift', 'Rotate hue by this fraction of the color wheel.', 'meaning', 'Changing colors can change a color-dependent class or attribute.'),
    'grayscale': _row('Grayscale blend', 'Blend toward grayscale by this fraction.', 'meaning', 'Removing color can change a color-dependent class or attribute.'),
    'invert': _row('Color inversion', 'Blend toward inverted colors by this fraction.', 'meaning', 'Inverting colors can change class-defining colors and contrast.'),
    'posterize_bits': _row('Posterization', 'Keep this many bits per color channel.', 'visibility', 'Color quantization can remove visual evidence.'),
    'solarize_threshold': _row('Solarization', 'Invert channel values at or above the threshold; 256 is unchanged.', 'meaning', 'Solarization changes colors and can alter class-defining evidence.'),
    'translate_x': _row('Horizontal translation', 'Shift by this fraction of image width; uncovered pixels are black.', 'visibility', 'Translation can crop the labeled object out of the frame.'),
    'translate_y': _row('Vertical translation', 'Shift by this fraction of image height; uncovered pixels are black.', 'visibility', 'Translation can crop the labeled object out of the frame.'),
    'zoom': _row('Centered zoom', 'Scale around the image center, retaining the canvas size.', 'visibility', 'Zoom can crop objects or make them too small to identify.'),
    'horizontal_flip': _row('Horizontal flip', 'Mirror left and right when set to 1.', 'meaning', 'Mirroring changes text, handedness, and directional labels.'),
}
_TEXT_METADATA = {
    'drop_stop_words': _row('Drop stop words', 'Remove common function words.', 'meaning', 'Removing function words can change relationships or attribution.'),
    'remove_punctuation': _row('Remove punctuation', 'Remove ASCII punctuation.', 'meaning', 'Punctuation removal can change numbers, ranges, quotations, and attribution.'),
    'to_lowercase': _row('Lowercase', 'Convert text to lowercase.'),
    'negate_sentence': _row('Negate a sentence', 'Remove the first not, or negate a supported auxiliary verb.', 'meaning', 'Negation deliberately changes the proposition; the original target may no longer apply.'),
    'delete_random_character': _row('Delete a character', 'Delete one interior character in an eligible word.', 'meaning', 'Character deletion can change names, quantities, or words.'),
    'insert_random_character': _row('Insert a character', 'Insert one seeded random letter.', 'meaning', 'Character insertion can change names or words.'),
    'change_letter_case': _row('Swap word case', 'Swap the case of one seeded word.'),
    'apply_misspellings': _row('Common misspellings', 'Apply a fixed table of common spelling mistakes.', 'meaning', 'Misspellings can change or obscure source words.'),
    'repeat_random_character': _row('Repeat a character', 'Repeat one interior character.', 'meaning', 'Character repetition can alter names, quantities, or words.'),
    'replace_with_adjacent_character': _row('Keyboard neighbor', 'Replace one letter with an adjacent keyboard key.', 'meaning', 'Keyboard substitutions can alter names or words.'),
    'swap_random_character': _row('Swap characters', 'Swap adjacent characters in an eligible word.', 'meaning', 'Character swaps can alter names, quantities, or words.'),
    'delete_random_word': _row('Delete a word', 'Delete one seeded word when several are present.', 'meaning', 'Removing a word can remove a fact, entity, number, or negation.'),
    'perturb_word_order': _row('Swap word order', 'Swap one word with its neighbor.', 'meaning', 'Word order changes can reverse relationships or attribution.'),
    'repeat_random_word': _row('Repeat a word', 'Repeat one seeded word.', 'meaning', 'Repeated words can change emphasis or interpretation.'),
    'to_uppercase': _row('Uppercase', 'Convert text to uppercase.'),
    'to_titlecase': _row('Title case', 'Capitalize each word.', 'structure', 'Case changes can affect acronyms and named entities.'),
    'normalize_whitespace': _row('Normalize whitespace', 'Collapse whitespace into single spaces.', 'structure', 'Flattening lines can remove dialogue speaker boundaries.'),
    'expand_contractions': _row('Expand contractions', 'Expand a fixed table of negative contractions.'),
    'contract_phrases': _row('Contract phrases', 'Contract a fixed table of negative verb phrases.'),
    'unicode_quotes': _row('Curly quotes', 'Replace paired straight quotes and apostrophes with curly forms.'),
    'ascii_quotes': _row('Straight quotes', 'Replace curly quotes and apostrophes with ASCII forms.'),
    'add_line_breaks': _row('Sentence line breaks', 'Put each punctuation-delimited sentence on its own line.', 'structure', 'Sentence segmentation is heuristic and may change dialogue structure.'),
    'add_sentence_numbers': _row('Number sentences', 'Add a numbered prefix to each sentence.', 'structure', 'Added numbers and segmentation may be mistaken for source content.'),
    'bullet_sentences': _row('Bullet sentences', 'Present punctuation-delimited sentences as a bullet list.', 'structure', 'List formatting can change discourse structure.'),
    'reverse_sentences': _row('Reverse sentences', 'Reverse punctuation-delimited sentence order.', 'meaning', 'Sentence order can change chronology and pronoun references.'),
    'duplicate_sentence': _row('Duplicate a sentence', 'Repeat one seeded sentence.', 'meaning', 'Duplicating a sentence can change emphasis or event counts.'),
    'truncate_last_sentence': _row('Remove final sentence', 'Remove the final sentence when several are present.', 'meaning', 'Removing a sentence removes source evidence; the original target may no longer apply.'),
    'replace_spaces_with_nbsp': _row('Nonbreaking spaces', 'Replace ordinary spaces with nonbreaking spaces.'),
}
_GENERATION_METADATA = {
    'style': _row('Style', 'Add a visual medium instruction.'),
    'lighting': _row('Lighting', 'Add a lighting instruction.'),
    'detail': _row('Detail level', 'Add a level-of-detail instruction.'),
    'composition': _row('Composition', 'Add a composition instruction.', 'structure', 'Composition may conflict with the scene’s specified spatial arrangement.'),
    'viewpoint': _row('Viewpoint', 'Add a camera-view instruction.', 'visibility', 'A new viewpoint may hide requested content or conflict with scene constraints.'),
    'framing': _row('Framing', 'Add a wide, medium, or close-up framing instruction.', 'visibility', 'Framing may crop requested objects or conflict with scene constraints.'),
    'color_palette': _row('Color palette', 'Add a palette instruction.', 'meaning', 'A palette instruction may conflict with explicitly requested object colors.'),
    'focus': _row('Focus', 'Add a depth-of-field or focus instruction.', 'visibility', 'Focus instructions may obscure requested objects or details.'),
    'texture': _row('Texture', 'Add a surface texture instruction.', 'meaning', 'Texture instructions may conflict with specified materials or surfaces.'),
}


def available_action_space(task):
    if task == 'classification':
        return copy.deepcopy(IMAGE_CATALOG)
    if task == 'summarization':
        return {'actions': list(TEXT_ACTIONS)}
    if task == 'generation':
        return copy.deepcopy(GENERATION_DIMENSIONS)
    raise ValueError('Unknown task.')


def action_metadata(task):
    if task == 'classification':
        result = copy.deepcopy(_IMAGE_METADATA)
        for key, row in result.items():
            low, high, identity = IMAGE_DIMENSIONS[key]
            row.update(minimum=low, maximum=high, identity=identity,
                       integer=key in INTEGER_IMAGE_DIMENSIONS, optional=key not in DEFAULT_IMAGE_SPACE)
        return result
    if task == 'summarization':
        return {name: dict(row, optional=name not in DEFAULT_TEXT_ACTIONS) for name, row in _TEXT_METADATA.items()}
    if task == 'generation':
        return {name: dict(row, optional=name not in DEFAULT_GENERATION_DIMENSIONS) for name, row in _GENERATION_METADATA.items()}
    raise ValueError('Unknown task.')


def action_semantic_risks(task, action):
    """Review hints for active nonidentity edits; these are not failure verdicts."""
    metadata = action_metadata(task)
    active = {'name': action.get('name')} if task == 'summarization' else action
    names = [active['name']] if task == 'summarization' else [key for key in active if key != 'id']
    if any(name not in metadata for name in names):
        raise ValueError('Unknown action in semantic review.')
    risks = []
    for name in metadata:
        if name not in names:
            continue
        row = metadata[name]
        risk, reason = row['semantic_risk'], row['risk_reason']
        if task == 'classification':
            value = action[name]
            if value == IMAGE_DIMENSIONS[name][2]:
                continue
            if ((name == 'rotation' and abs(value) > 30) or
                    (name == 'darkness' and value > .7) or
                    (name == 'saturation' and (value == 0 or value > 2))):
                risk, reason = 'visibility', 'This strong transformation may remove or alter class-defining visual evidence.'
        elif task == 'generation' and action[name] is None:
            continue
        if risk != 'low':
            risks.append(dict(name=name, label=row['label'], semantic_risk=risk, reason=reason))
    return risks

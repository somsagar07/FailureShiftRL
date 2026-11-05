"""Small, deterministic search primitives; importing this module loads no model."""
import itertools
import io
import math
import random
import re
import string
from collections import Counter

from action_catalog import (DEFAULT_IMAGE_SPACE, DEFAULT_GENERATION_DIMENSIONS, DEFAULT_TEXT_ACTIONS,
                            GENERATION_STYLES, GENERATION_DIMENSIONS, IMAGE_DIMENSIONS,
                            INTEGER_IMAGE_DIMENSIONS, TEXT_ACTIONS, available_action_space,
                            action_metadata, action_semantic_risks)


MAX_ACTIONS = 5000
LEGACY_GENERATION_SPACE = dict(description=['unique', 'distinctive', 'cool'],
                               profession=['scientist', 'artist', 'professor'],
                               setting=['corporate office', 'research center', 'classroom'])


def default_action_space(task):
    if task == 'classification':
        # Same order and values as Experiments/Classification/config.py.
        return {key: list(values) for key, values in DEFAULT_IMAGE_SPACE.items()}
    if task == 'summarization':
        return {'actions': list(DEFAULT_TEXT_ACTIONS)}
    if task == 'generation':
        return {key:list(values) for key,values in DEFAULT_GENERATION_DIMENSIONS.items()}
    raise ValueError('Unknown task.')


def normalize_action_space(task, space=None):
    defaults = default_action_space(task)
    if space is None:
        space = defaults
    elif task == 'generation' and isinstance(space, dict) and set(space) == set(LEGACY_GENERATION_SPACE):
        # Preserve explicitly saved template spaces without making them new defaults.
        defaults = LEGACY_GENERATION_SPACE
    elif task == 'generation' and isinstance(space, dict) and set(space) == {'styles'}:
        defaults = {'styles': GENERATION_STYLES}
    grouped_generation = task == 'generation' and set(defaults) == set(DEFAULT_GENERATION_DIMENSIONS)
    allowed = (IMAGE_DIMENSIONS if task == 'classification' else
               GENERATION_DIMENSIONS if grouped_generation else defaults)
    if not isinstance(space, dict) or not set(defaults) <= set(space) or set(space) - set(allowed):
        raise ValueError('Action space must contain: ' + ', '.join(defaults) + '; use only available optional groups.')
    allow_disabled = task == 'classification' or grouped_generation
    normalized = {}
    for key in allowed:
        if key not in space:
            continue
        values = space[key]
        limit = len(TEXT_ACTIONS) if task == 'summarization' else 25
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f'{key}: use a list of at most {limit} values.')
        cleaned = []
        for value in values:
            if task == 'classification':
                _validate_image_value(key, value)
            else:
                if not isinstance(value, str) or not value.strip() or len(value) > 80 or any(ord(c) < 32 for c in value):
                    raise ValueError(f'{key}: enter nonempty text values of up to 80 characters.')
                value = value.strip()
                if task == 'summarization' and value not in TEXT_ACTIONS:
                    raise ValueError(f'Unknown text action: {value}.')
                if task == 'generation' and key == 'styles' and value not in GENERATION_STYLES:
                    raise ValueError(f'Unknown generation style: {value}. Choose an available style.')
                if grouped_generation and value not in GENERATION_DIMENSIONS[key]:
                    raise ValueError(f'Unknown generation {key}: {value}. Choose an available value.')
            if value not in cleaned:
                cleaned.append(value)
        if not cleaned and not allow_disabled:
            raise ValueError(f'Choose at least one value for {key}.')
        normalized[key] = cleaned
    if allow_disabled and not any(normalized.values()):
        raise ValueError('Enable at least one transformation in the action space.')
    size = len(normalized['actions']) if task == 'summarization' else math.prod(max(1, len(v)) for v in normalized.values())
    if size > MAX_ACTIONS:
        raise ValueError(f'Action space has {size} combinations. Use at most {MAX_ACTIONS}.')
    return normalized


def actions_for(task, space=None):
    space = normalize_action_space(task, space)
    if task == 'summarization':
        # IDs stay stable when other actions are removed, preserving per-action RNG.
        return [dict(id=TEXT_ACTIONS.index(name), name=name) for name in space['actions']]
    if task == 'generation' and 'styles' in space:
        return [dict(id=GENERATION_STYLES.index(style), style=style) for style in space['styles']]
    if task == 'generation' and set(space) <= set(GENERATION_DIMENSIONS):
        # Keep the original three digits in their original positions. Optional
        # axes occupy higher digits, so saved IDs do not change as catalog grows.
        keys = [key for key in GENERATION_DIMENSIONS if key in DEFAULT_GENERATION_DIMENSIONS or space.get(key)]
        actions = []
        for combination in itertools.product(*(space[key] or [None] for key in keys)):
            # Reserve zero in each axis for an omitted instruction. IDs remain
            # stable when values are filtered or another group is removed.
            identity = 0
            chosen = dict(zip(keys, combination))
            for key, values in DEFAULT_GENERATION_DIMENSIONS.items():
                value = chosen[key]
                identity = identity * (len(values)+1) + (values.index(value)+1 if value is not None else 0)
            multiplier = math.prod(len(values)+1 for values in DEFAULT_GENERATION_DIMENSIONS.values())
            for key, values in GENERATION_DIMENSIONS.items():
                if key in DEFAULT_GENERATION_DIMENSIONS:
                    continue
                value = chosen.get(key)
                identity += multiplier * (values.index(value)+1 if value is not None else 0)
                multiplier *= len(values)+1
            actions.append(dict(id=identity, **dict(zip(keys, combination))))
        return actions
    keys = (list(LEGACY_GENERATION_SPACE) if task == 'generation' else
            [key for key in IMAGE_DIMENSIONS if key in DEFAULT_IMAGE_SPACE or space.get(key)])
    values = [space[key] or [IMAGE_DIMENSIONS[key][2]] for key in keys]
    return [dict(id=i, **dict(zip(keys, combination))) for i, combination in enumerate(itertools.product(*values))]


def image_actions():
    return actions_for('classification')


def text_actions():
    return actions_for('summarization')


def generation_actions():
    return actions_for('generation')


def select_actions(actions, budget, seed, strategy):
    selected = list(actions)
    if strategy == 'random':
        random.Random(seed).shuffle(selected)
    return selected[:max(0, budget)]


def classification_outcome(baseline_id, predicted_id, expected_id):
    if expected_id is not None:
        if baseline_id != expected_id:
            return 'baseline_error'
        return 'label_violation' if predicted_id != expected_id else 'stable'
    return 'candidate' if predicted_id != baseline_id else 'stable'


def apply_image(image, action):
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    if not isinstance(action, dict) or set(action) - set(IMAGE_DIMENSIONS) - {'id'}:
        raise ValueError('Choose only available image transformations.')
    for name, value in action.items():
        if name != 'id':
            _validate_image_value(name, value)
    values = {name: action.get(name, bounds[2]) for name, bounds in IMAGE_DIMENSIONS.items()}
    if all(values[name] == bounds[2] for name, bounds in IMAGE_DIMENSIONS.items()):
        return image.copy()
    original_mode = image.mode
    result = image if image.mode in ('RGB', 'RGBA', 'L', 'LA') else image.convert('RGB')
    # Research pipeline: rotate -> saturation -> darken, on a 224px square.
    result = result.rotate(values['rotation'])
    result = ImageEnhance.Color(result).enhance(values['saturation'])
    result = ImageEnhance.Brightness(result).enhance(1 - values['darkness'])
    # Fixed order is independent of JSON ordering and catalog selection. Alpha
    # is retained for photometric edits and transformed with spatial edits.
    for name, enhancer in [('brightness', ImageEnhance.Brightness), ('contrast', ImageEnhance.Contrast),
                           ('sharpness', ImageEnhance.Sharpness)]:
        if values[name] != 1:
            result = enhancer(result).enhance(values[name])
    if values['gaussian_blur']:
        result = _map_image_color(result, lambda color: color.filter(ImageFilter.GaussianBlur(values['gaussian_blur'])))
    if values['gaussian_noise']:
        import numpy as np
        def noise(color):
            pixels = np.asarray(color, dtype=np.float32)
            perturbation = np.random.default_rng(42).normal(0, values['gaussian_noise']*255, pixels.shape)
            return Image.fromarray(np.clip(pixels+perturbation, 0, 255).astype(np.uint8))
        result = _map_image_color(result, noise)
    if values['jpeg_quality'] != 100:
        def compress(color):
            with io.BytesIO() as buffer:
                color.save(buffer, format='JPEG', quality=int(values['jpeg_quality']))
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    return decoded.convert(color.mode)
        result = _map_image_color(result, compress)
    if values['pixelation'] != 1:
        factor = values['pixelation']
        size = (max(1, round(result.width/factor)), max(1, round(result.height/factor)))
        result = result.resize(size, Image.Resampling.BOX).resize(result.size, Image.Resampling.NEAREST)
    if values['hue_shift']:
        import numpy as np
        def shift_hue(color):
            if color.mode == 'L':
                return color.copy()
            hsv = np.asarray(color.convert('HSV')).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16)+round(values['hue_shift']*256)) % 256
            return Image.frombytes('HSV', color.size, hsv.tobytes()).convert('RGB')
        result = _map_image_color(result, shift_hue)
    if values['grayscale']:
        result = ImageEnhance.Color(result).enhance(1-values['grayscale'])
    if values['invert']:
        result = _map_image_color(result, lambda color: Image.blend(color, ImageOps.invert(color), values['invert']))
    if values['posterize_bits'] != 8:
        result = _map_image_color(result, lambda color: ImageOps.posterize(color, int(values['posterize_bits'])))
    if values['solarize_threshold'] != 256:
        result = _map_image_color(result, lambda color: ImageOps.solarize(color, int(values['solarize_threshold'])))
    if values['translate_x'] or values['translate_y']:
        offset = (1, 0, -values['translate_x']*result.width, 0, 1, -values['translate_y']*result.height)
        result = result.transform(result.size, Image.Transform.AFFINE, offset, Image.Resampling.BICUBIC)
    if values['zoom'] != 1:
        scale = 1/values['zoom']
        matrix = (scale, 0, (result.width-1)*(1-scale)/2, 0, scale, (result.height-1)*(1-scale)/2)
        result = result.transform(result.size, Image.Transform.AFFINE, matrix, Image.Resampling.BICUBIC)
    if values['horizontal_flip']:
        result = ImageOps.mirror(result)
    return result if result.mode == original_mode else result.convert(original_mode)


def _validate_image_value(name, value):
    low, high, _ = IMAGE_DIMENSIONS[name]
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f'{name}: values must be numbers between {low} and {high}.')
    if name in INTEGER_IMAGE_DIMENSIONS and value != int(value):
        raise ValueError(f'{name}: use whole-number values between {low} and {high}.')


def _map_image_color(image, operation):
    if image.mode not in ('RGBA', 'LA'):
        return operation(image)
    result = operation(image.convert('RGB' if image.mode == 'RGBA' else 'L'))
    result.putalpha(image.getchannel('A'))
    return result


def apply_prompt(text, action):
    """Apply presentation settings without rewriting the original scene text."""
    if not isinstance(action, dict):
        raise ValueError('Choose an available generation action.')
    if any(key in action for key in GENERATION_DIMENSIONS):
        if set(action) - set(GENERATION_DIMENSIONS) - {'id'}:
            raise ValueError('Choose only available generation presentation groups.')
        instructions = []
        for key, values in GENERATION_DIMENSIONS.items():
            value = action.get(key)
            if value is None:
                continue
            if value not in values:
                raise ValueError(f'Choose an available generation {key}.')
            if key in ('style', 'lighting'):
                instructions.append(f'{value[0].upper()}{value[1:]} {key}.')
            else:
                label = 'Detail level' if key == 'detail' else key.replace('_', ' ').capitalize()
                instructions.append(f'{label}: {value}.')
        if not instructions:
            raise ValueError('Enable at least one presentation instruction.')
        return ' '.join(instructions) + ' Scene: ' + text
    # Only explicit legacy jobs substitute template slots; other braces stay literal.
    if set(action) - set(LEGACY_GENERATION_SPACE) - {'id'}:
        raise ValueError('Choose only available generation template fields.')
    for key in LEGACY_GENERATION_SPACE:
        if key not in action:
            raise ValueError('Choose a generation style or a complete legacy template action.')
        text = text.replace('{' + key + '}', action[key])
    return text


def apply_text(text, action, rng):
    """Bounded variants of the original named actions; never spin on short text."""
    name = action.get('name') if isinstance(action, dict) else None
    if name not in TEXT_ACTIONS:
        raise ValueError(f'Unknown text action: {name}.')
    if name == 'to_uppercase':
        return text.upper()
    if name == 'to_titlecase':
        return text.title()
    if name == 'normalize_whitespace':
        return ' '.join(text.split())
    if name in ('expand_contractions', 'contract_phrases'):
        return _replace_contractions(text, expand=name == 'expand_contractions')
    if name == 'unicode_quotes':
        return re.sub(r'"([^"]*)"', lambda match: '“'+match[1]+'”', text).replace("'", '’')
    if name == 'ascii_quotes':
        return text.translate(str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"}))
    if name == 'replace_spaces_with_nbsp':
        return text.replace(' ', '\u00a0')
    if name in ('add_line_breaks', 'add_sentence_numbers', 'bullet_sentences',
                'reverse_sentences', 'duplicate_sentence', 'truncate_last_sentence'):
        if not text.strip():
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if name == 'add_line_breaks':
            return '\n'.join(sentences)
        if name == 'add_sentence_numbers':
            return '\n'.join(f'{index}. {sentence}' for index, sentence in enumerate(sentences, 1))
        if name == 'bullet_sentences':
            return '\n'.join('- '+sentence for sentence in sentences)
        if name == 'reverse_sentences':
            return ' '.join(reversed(sentences))
        if name == 'duplicate_sentence':
            index = rng.randrange(len(sentences))
            sentences.insert(index, sentences[index])
        elif len(sentences) > 1:
            sentences.pop()
        return ' '.join(sentences)
    words = text.split()
    if name == 'remove_punctuation':
        return text.translate(str.maketrans('', '', string.punctuation))
    if name == 'to_lowercase':
        return text.lower()
    if not words:
        return text
    if name == 'drop_stop_words':
        stop = set('a an the and or but of to in on at for with from by is are was were be been it its this that as'.split())
        return ' '.join(w for w in words if w.lower().strip(string.punctuation) not in stop)
    if name == 'negate_sentence':
        match = re.search(r'\bnot\s+', text, re.I)
        if match:
            return text[:match.start()] + text[match.end():]
        return re.sub(r'\b(is|are|was|were|can|will|has|have)\b', r'\1 not', text, count=1, flags=re.I)
    i = rng.randrange(len(words))
    if name == 'delete_random_word':
        if len(words) > 1:
            words.pop(i)
    elif name == 'repeat_random_word':
        words.insert(i, words[i])
    elif name == 'perturb_word_order':
        if len(words) > 1:
            j = (i + 1) % len(words)
            words[i], words[j] = words[j], words[i]
    elif name == 'change_letter_case':
        words[i] = words[i].swapcase()
    elif name == 'apply_misspellings':
        changes = {'the': 'teh', 'their': 'thier', 'because': 'becuase', 'people': 'poeple', 'research': 'reserach', 'with': 'wiht'}
        words = [changes.get(w.lower(), w) for w in words]
    else:
        eligible = [j for j, word in enumerate(words) if len(word) > 2 and any(c.isalpha() for c in word)]
        if not eligible:
            return text
        i = rng.choice(eligible)
        word = words[i]
        j = rng.randrange(1, len(word) - 1)
        if name == 'delete_random_character':
            word = word[:j] + word[j+1:]
        elif name == 'insert_random_character':
            word = word[:j] + rng.choice(string.ascii_lowercase) + word[j:]
        elif name == 'repeat_random_character':
            word = word[:j] + word[j] + word[j:]
        elif name == 'replace_with_adjacent_character':
            rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
            for row in rows:
                if word[j].lower() in row:
                    pos = row.index(word[j].lower())
                    replacement = row[pos-1] if pos else row[1]
                    word = word[:j] + replacement + word[j+1:]
                    break
        elif name == 'swap_random_character':
            word = word[:j] + word[j+1] + word[j] + word[j+2:]
        words[i] = word
    return ' '.join(words)


def _replace_contractions(text, *, expand):
    pairs = {"can't": 'cannot', "won't": 'will not', "don't": 'do not', "doesn't": 'does not',
             "didn't": 'did not', "isn't": 'is not', "aren't": 'are not', "wasn't": 'was not',
             "weren't": 'were not', "hasn't": 'has not', "haven't": 'have not', "hadn't": 'had not',
             "couldn't": 'could not', "shouldn't": 'should not', "wouldn't": 'would not', "mustn't": 'must not'}
    replacements = pairs if expand else {value: key for key, value in pairs.items()}
    def replace(match):
        source = match[0]
        value = replacements[source.lower()]
        if source.isupper():
            return value.upper()
        return value[0].upper()+value[1:] if source[0].isupper() else value
    return re.sub(r'\b(?:'+'|'.join(re.escape(key) for key in replacements)+r')\b', replace, text, flags=re.I)


def rouge_l(prediction, reference):
    """Lowercase word-token ROUGE-L F1; a proxy, not a factuality oracle."""
    a, b = re.findall(r'\w+', prediction.lower()), re.findall(r'\w+', reference.lower())
    if not a or not b:
        return 0.0
    row = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            old = row[j]
            row[j] = previous + 1 if x == y else max(row[j], row[j-1])
            previous = old
    return 2 * row[-1] / (len(a) + len(b))


def counts(results):
    return dict(Counter(row['status'] for row in results))

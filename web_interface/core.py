"""Small, deterministic search primitives; importing this module loads no model."""
import itertools
import math
import random
import re
import string
from collections import Counter


MAX_ACTIONS = 500
IMAGE_DIMENSIONS = {'rotation': (-180, 180, 0), 'darkness': (0, 1, 0), 'saturation': (0, 3, 1)}
GENERATION_STYLES = ['photograph', 'watercolor', 'pencil drawing', 'digital illustration', '3D render']
GENERATION_DIMENSIONS = dict(style=GENERATION_STYLES,
    lighting=['even', 'soft', 'directional', 'high-key', 'low-key'],
    detail=['minimal', 'simple', 'moderate', 'detailed', 'intricate'])
LEGACY_GENERATION_SPACE = dict(description=['unique', 'distinctive', 'cool'],
                               profession=['scientist', 'artist', 'professor'],
                               setting=['corporate office', 'research center', 'classroom'])


TEXT_ACTIONS = [
    'drop_stop_words', 'remove_punctuation', 'to_lowercase', 'negate_sentence',
    'delete_random_character', 'insert_random_character', 'change_letter_case',
    'apply_misspellings', 'repeat_random_character', 'replace_with_adjacent_character',
    'swap_random_character', 'delete_random_word', 'perturb_word_order', 'repeat_random_word',
]


def default_action_space(task):
    if task == 'classification':
        # Same order and values as Experiments/Classification/config.py.
        return dict(rotation=[-10, -5, 0, 5, 10], darkness=[.1, .2, .3, .4, .5], saturation=[.5, .55, .6, .65, .7])
    if task == 'summarization':
        return {'actions': list(TEXT_ACTIONS)}
    if task == 'generation':
        return {key:list(values) for key,values in GENERATION_DIMENSIONS.items()}
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
    if not isinstance(space, dict) or set(space) != set(defaults):
        raise ValueError('Action space must contain: ' + ', '.join(defaults) + '.')
    grouped_generation = task == 'generation' and set(defaults) == set(GENERATION_DIMENSIONS)
    allow_disabled = task == 'classification' or grouped_generation
    normalized = {}
    for key, values in space.items():
        if not isinstance(values, list) or len(values) > 25:
            raise ValueError(f'{key}: use a list of at most 25 values.')
        cleaned = []
        for value in values:
            if task == 'classification':
                low, high, _ = IMAGE_DIMENSIONS[key]
                if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f'{key}: values must be numbers between {low} and {high}.')
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
    if task == 'generation' and set(space) == set(GENERATION_DIMENSIONS):
        keys = list(GENERATION_DIMENSIONS)
        actions = []
        for combination in itertools.product(*(space[key] or [None] for key in keys)):
            # Reserve zero in each axis for an omitted instruction. IDs remain
            # stable when values are filtered or another group is removed.
            identity = 0
            for key, value in zip(keys, combination):
                values = GENERATION_DIMENSIONS[key]
                identity = identity * (len(values)+1) + (values.index(value)+1 if value is not None else 0)
            actions.append(dict(id=identity, **dict(zip(keys, combination))))
        return actions
    keys = list(LEGACY_GENERATION_SPACE if task == 'generation' else default_action_space(task))
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
    from PIL import ImageEnhance
    # Research pipeline: rotate -> saturation -> darken, on a 224px square.
    result = image.rotate(action['rotation'])
    result = ImageEnhance.Color(result).enhance(action['saturation'])
    return ImageEnhance.Brightness(result).enhance(1 - action['darkness'])


def apply_prompt(text, action):
    """Apply presentation settings without rewriting the original scene text."""
    if any(key in action for key in GENERATION_DIMENSIONS):
        instructions = []
        for key, values in GENERATION_DIMENSIONS.items():
            value = action.get(key)
            if value is None:
                continue
            if value not in values:
                raise ValueError(f'Choose an available generation {key}.')
            instructions.append(f'Detail level: {value}.' if key == 'detail' else f'{value[0].upper()}{value[1:]} {key}.')
        if not instructions:
            raise ValueError('Enable at least one presentation instruction.')
        return ' '.join(instructions) + ' Scene: ' + text
    # Only explicit legacy jobs substitute template slots; other braces stay literal.
    for key in LEGACY_GENERATION_SPACE:
        if key not in action:
            raise ValueError('Choose a generation style or a complete legacy template action.')
        text = text.replace('{' + key + '}', action[key])
    return text


def apply_text(text, action, rng):
    """Bounded variants of the original named actions; never spin on short text."""
    name = action['name']
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

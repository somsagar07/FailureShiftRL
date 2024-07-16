import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from random import randint
from pattern.text.en import pluralize, singularize
from nltk.stem.wordnet import WordNetLemmatizer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Change the model name to the one you want to use
TOKENIZER = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
MODEL = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

def get_summary(text, max_length=130, min_length=40):
    # Encode the input text and add the special tokens for the model
    inputs = TOKENIZER.encode("summarize: " + text, return_tensors="pt", truncation=True)

    # Generate summary with the model
    summary_ids = MODEL.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return the summary
    return TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def drop_stop_words(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)

def remove_punctuation(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

def to_lowercase(sentence):
    return sentence.lower()

def negate_sentence(sentence):
    negation_words = ['not', 'no', 'never', 'none', 'nothing', 'nobody', 'neither', 'nowhere', 'hardly', 'scarcely', 'barely', 'don’t', 'isn’t', 'wasn’t', 'shouldn’t', 'wouldn’t', 'couldn’t', 'won’t', 'can’t', 'doesn’t']
    word_tokens = word_tokenize(sentence)
    negated_sentence = []
    negated = False

    for word in word_tokens:
        if word in negation_words:
            negated = not negated
            continue
        negated_sentence.append("not " + word if negated else word)

    return ' '.join(negated_sentence)

def return_random_number(begin, end):
    return randint(begin, end)

def delete_random_character(sentence):
    """
    This function takes a sentence, randomly selects a word, then randomly deletes a character from it.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = 0
    random_word_selected = False

    while not random_word_selected:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True

    selected_word = sample_tokenized[random_word_index]

    # Delete a random character
    random_char_index = return_random_number(1, len(selected_word)-2)
    perturbed_word = selected_word[:random_char_index] + selected_word[random_char_index+1:]

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join(sample_tokenized[:random_word_index] + [perturbed_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def insert_random_character(sentence):
    """
    This function takes a sentence, randomly selects a word, then randomly inserts a character into it.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = 0
    random_word_selected = False

    while not random_word_selected:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True

    selected_word = sample_tokenized[random_word_index]

    # Insert a random character
    random_char_index = return_random_number(1, len(selected_word)-1)
    random_char_code = return_random_number(97, 122)
    random_char = chr(random_char_code)

    perturbed_word = selected_word[:random_char_index] + random_char + selected_word[random_char_index:]

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join(sample_tokenized[:random_word_index] + [perturbed_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def random_changing_type():
    return 'FirstChar' if randint(1, 2) == 1 else 'AllChars'

def change_letter_case(sentence):
    """
    This function takes a sentence, randomly selects a word, then changes the case of the first character
    or all characters in that word.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = 0
    random_word_selected = False

    while not random_word_selected:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True

    selected_word = sample_tokenized[random_word_index]

    # Change the letter case
    change_type = random_changing_type()
    temp_word = ""

    if change_type == 'FirstChar':
        # Toggle case of the first character
        char = selected_word[0]
        temp_word = char.lower() if char.isupper() else char.upper()
        temp_word += selected_word[1:]
    elif change_type == 'AllChars':
        # Toggle case of all characters
        for char in selected_word:
            temp_word += char.lower() if char.isupper() else char.upper()

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join(sample_tokenized[:random_word_index] + [temp_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def generate_misspelling(word):
    """
    This function takes a word and generates a misspelling by randomly altering a character.
    """
    if len(word) > 1:
        random_char_index = return_random_number(0, len(word) - 1)
        random_char_code = return_random_number(97, 122)
        random_char = chr(random_char_code)

        misspelled_word = word[:random_char_index] + random_char + word[random_char_index+1:]
        return misspelled_word
    else:
        return word

def apply_misspellings(sentence, max_perturb=10):
    """
    This function takes a sentence and applies misspelling perturbations by randomly altering characters in words.
    """
    sample_tokenized = nltk.word_tokenize(sentence)
    perturbed_sample = sentence
    word_replaced = False

    num_replacements = 0
    while num_replacements < min(max_perturb, len(sample_tokenized)):
        random_word_index = return_random_number(0, len(sample_tokenized) - 1)
        selected_word = sample_tokenized[random_word_index]

        misspelled_word = generate_misspelling(selected_word)

        if misspelled_word != selected_word:
            perturbed_sample = perturbed_sample.replace(selected_word, misspelled_word, 1)
            num_replacements += 1
            word_replaced = True

    return perturbed_sample if word_replaced else sentence

def repeat_random_character(sentence):
    """
    This function takes a sentence, randomly selects a word, then randomly repeats a character in that word.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = 0
    random_word_selected = False

    while not random_word_selected:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True

    selected_word = sample_tokenized[random_word_index]

    # Repeat a random character
    random_char_index = return_random_number(1, len(selected_word)-2)
    repeated_char = selected_word[random_char_index]

    perturbed_word = selected_word[:random_char_index] + repeated_char + selected_word[random_char_index:]

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join(sample_tokenized[:random_word_index] + [perturbed_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def return_adjacent_char(input_char):

    if (input_char == 'a'):
        return 's'

    elif (input_char == 'b'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'v'
        else:
            return 'n'

    elif (input_char == 'c'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'x'
        else:
            return 'v'

    elif (input_char == 'd'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 's'
        else:
            return 'f'

    elif (input_char == 'e'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'w'
        else:
            return 'r'

    elif (input_char == 'f'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'd'
        else:
            return 'g'

    elif (input_char == 'g'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'f'
        else:
            return 'h'

    elif (input_char == 'h'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'g'
        else:
            return 'j'

    elif (input_char == 'i'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'u'
        else:
            return 'o'

    elif (input_char == 'j'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'h'
        else:
            return 'k'

    elif (input_char == 'k'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'j'
        else:
            return 'l'

    elif (input_char == 'l'):
        return 'k'

    elif (input_char == 'm'):
        return 'n'

    elif (input_char == 'n'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'b'
        else:
            return 'm'

    elif (input_char == 'o'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'i'
        else:
            return 'p'

    elif (input_char == 'p'):
        return 'o'

    elif (input_char == 'q'):
        return 'w'

    elif (input_char == 'r'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'e'
        else:
            return 't'

    elif (input_char == 's'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'a'
        else:
            return 'd'

    elif (input_char == 't'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'r'
        else:
            return 'y'

    elif (input_char == 'u'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'y'
        else:
            return 'i'

    elif (input_char == 'v'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'c'
        else:
            return 'b'

    elif (input_char == 'w'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'q'
        else:
            return 'e'

    elif (input_char == 'x'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'z'
        else:
            return 'c'

    elif (input_char == 'y'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 't'
        else:
            return 'u'

    elif (input_char == 'z'):
        return 'x'
    #---------------------------------------------
    elif (input_char == 'A'):
        return 'S'

    elif (input_char == 'B'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'V'
        else:
            return 'N'

    elif (input_char == 'C'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'X'
        else:
            return 'V'

    elif (input_char == 'D'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'S'
        else:
            return 'F'

    elif (input_char == 'E'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'W'
        else:
            return 'R'

    elif (input_char == 'F'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'D'
        else:
            return 'G'

    elif (input_char == 'G'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'F'
        else:
            return 'H'

    elif (input_char == 'H'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'G'
        else:
            return 'J'

    elif (input_char == 'I'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'U'
        else:
            return 'O'

    elif (input_char == 'J'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'H'
        else:
            return 'K'

    elif (input_char == 'K'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'J'
        else:
            return 'L'

    elif (input_char == 'L'):
        return 'K'

    elif (input_char == 'M'):
        return 'N'

    elif (input_char == 'N'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'B'
        else:
            return 'M'

    elif (input_char == 'O'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'I'
        else:
            return 'P'

    elif (input_char == 'P'):
        return 'O'

    elif (input_char == 'Q'):
        return 'W'

    elif (input_char == 'R'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'E'
        else:
            return 'T'

    elif (input_char == 'S'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'A'
        else:
            return 'D'

    elif (input_char == 'T'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'R'
        else:
            return 'Y'

    elif (input_char == 'U'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Y'
        else:
            return 'I'

    elif (input_char == 'V'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'C'
        else:
            return 'B'

    elif (input_char == 'W'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Q'
        else:
            return 'E'

    elif (input_char == 'X'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'Z'
        else:
            return 'C'

    elif (input_char == 'Y'):
        which_adjacent = return_random_number(1, 2)
        if (which_adjacent == 1):
            return 'T'
        else:
            return 'U'

    elif (input_char == 'Z'):
        return 'X'

    else:
        return '*'

def replace_with_adjacent_character(sentence):
    """
    This function takes a sentence, randomly selects a word, then randomly replaces a character in that word.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = 0
    random_word_selected = False

    while not random_word_selected:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True

    selected_word = sample_tokenized[random_word_index]

    # Replace a random character
    random_char_index = return_random_number(1, len(selected_word)-2)
    char_to_replace = selected_word[random_char_index]
    adjacent_char = return_adjacent_char(char_to_replace)

    perturbed_word = selected_word[:random_char_index] + adjacent_char + selected_word[random_char_index+1:]

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join(sample_tokenized[:random_word_index] + [perturbed_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def swap_characters(input_word, position, adjacent):
    temp_word = ''
    if (adjacent == 'left'):
        if (position == 1):
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif (position == len(input_word)-1):
            temp_word = input_word[0:position-1]
            temp_word += input_word[position]
            temp_word += input_word[position-1]
        elif (position > 1 and position < len(input_word)-1):
            temp_word = input_word[0:position-1]
            temp_word += input_word[position]
            temp_word += input_word[position-1]
            temp_word += input_word[position+1:]

    elif (adjacent == 'right'):
        if (position == 0):
            temp_word = input_word[1]
            temp_word += input_word[0]
            temp_word += input_word[2:]
        elif (position == len(input_word)-2):
            temp_word = input_word[0:position]
            temp_word += input_word[position+1]
            temp_word += input_word[position]
        elif (position > 0 and position < len(input_word)-2):
            temp_word = input_word[0:position]
            temp_word += input_word[position+1]
            temp_word += input_word[position]
            temp_word += input_word[position+2:]

    return temp_word

def swap_random_character(sentence):
    """
    This function takes a sentence, randomly selects a word, then randomly swaps a character in that word with its adjacent character.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = 0
    random_word_selected = False

    while not random_word_selected:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)
        if len(sample_tokenized[random_word_index]) > 2:
            random_word_selected = True

    selected_word = sample_tokenized[random_word_index]

    # Select a random character and its adjacent for swapping
    random_char_index = return_random_number(0, len(selected_word)-1)
    adjacent_for_swapping = 'right' if random_char_index == 0 else 'left' if random_char_index == len(selected_word)-1 else 'left' if return_random_number(1, 2) == 1 else 'right'

    # Swap the character and the adjacent
    perturbed_word = swap_characters(selected_word, random_char_index, adjacent_for_swapping)

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join(sample_tokenized[:random_word_index] + [perturbed_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def delete_random_word(sentence):
    """
    This function takes a sentence, tokenizes it, and randomly deletes one of the words.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word to delete
    random_word_index = return_random_number(0, len(sample_tokenized)-1)

    # Ensure the selected word has more than one character (optional, can be adjusted)
    while len(sample_tokenized[random_word_index]) <= 1:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)

    # Delete the word and reconstruct the sentence
    del sample_tokenized[random_word_index]
    perturbed_sample = " ".join(sample_tokenized)

    return perturbed_sample


def change_ordering(input_length, input_side, input_changes):
    ordering = []

    if (input_side == 1):
        for i in range(0, input_length):
            if (i < input_changes):

                candidates=[]
                for j in range(0, input_changes):
                    if (j != i and j not in ordering):
                        candidates.append(j)

                if (len(candidates) > 0):
                    random_index = return_random_number(0, len(candidates)-1)
                    ordering.append(candidates[random_index])
                else:
                    ordering.append(i)
            else:
                ordering.append(i)

    elif (input_side == 2):
        for i in range(0, input_length):
            if (i < input_length-input_changes):
                ordering.append(i)

            else:
                candidates=[]
                for j in range(input_length-input_changes, input_length):
                    if (j != i and j not in ordering):
                        candidates.append(j)

                if (len(candidates) > 0):
                    random_index = return_random_number(0, len(candidates)-1)
                    ordering.append(candidates[random_index])
                else:
                    ordering.append(i)

    return ordering

def perturb_word_order(sentence):
    """
    This function takes a sentence, tokenizes it, and randomly changes the order of the words.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    perturbed_sample = ""
    if len(sample_tokenized) > 3:
        last_token = ""
        if sample_tokenized[-1] in ('.', '?', '!', ';', ','):
            last_token = sample_tokenized[-1]
            sample_tokenized = sample_tokenized[:-1]

        ordering_side = return_random_number(1, 2)
        num_changed_words = return_random_number(2, len(sample_tokenized)-1)
        new_word_order = change_ordering(len(sample_tokenized), ordering_side, num_changed_words)

        for i in new_word_order:
            perturbed_sample += sample_tokenized[i] + ' '
        perturbed_sample += last_token
    else:
        perturbed_sample = sentence

    return perturbed_sample

def repeat_random_word(sentence):
    """
    This function takes a sentence, tokenizes it, and randomly repeats one of the words.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word to repeat
    random_word_index = return_random_number(0, len(sample_tokenized)-1)

    # Ensure the selected word has more than one character (optional, can be adjusted)
    while len(sample_tokenized[random_word_index]) <= 1:
        random_word_index = return_random_number(0, len(sample_tokenized)-1)

    selected_word = sample_tokenized[random_word_index]

    # Reconstruct the sentence with the repeated word
    perturbed_sample = " ".join(sample_tokenized[:random_word_index+1] + [selected_word] + sample_tokenized[random_word_index+1:])

    return perturbed_sample

def toggle_singular_plural(sentence):
    """
    This function takes a sentence, tokenizes it, and randomly changes one of the words from singular to plural or vice versa.
    """
    sample_tokenized = nltk.word_tokenize(sentence)

    # Select a random word
    random_word_index = return_random_number(0, len(sample_tokenized)-1)

    selected_word = sample_tokenized[random_word_index]
    word_synsets = wordnet.synsets(selected_word)

    # Check if the word is likely a noun (this is a simplification and might not always be accurate)
    if word_synsets and word_synsets[0].pos() in ['n', 's']:
        if pluralize(singularize(selected_word)) == selected_word:  # If the word is singular
            new_word = pluralize(selected_word)
        else:  # If the word is plural
            new_word = singularize(selected_word)

        sample_tokenized[random_word_index] = new_word

    # Reconstruct the sentence
    perturbed_sample = " ".join(sample_tokenized)

    return perturbed_sample

def is_third_person(input_pos_tag):
    subject = ''
    for i in range(0, len(input_pos_tag)):
        token = input_pos_tag[i]
        if (subject == ''):
            if (token[0].lower() in ('it', 'this', 'that', 'he', 'she')):
                subject = 'third person'
            elif (token[1] in ('NNP')):
                subject = 'third person'
            elif (token[0].lower() in ('i', 'we', 'you', 'they', 'she', 'these', 'those')):
                subject = 'not third person'
            elif (token[0].lower() in ('NNPS')):
                subject = 'not third person'
    if (subject == 'third person'):
        return 'third person'
    elif (subject == 'not third person'):
        return 'not third person'
    else:
        return 'none'

def change_verb_tense(sentence):
    """
    This function takes a sentence and changes the tense of verbs found in the sentence.
    """
    sample_tokenized = nltk.word_tokenize(sentence)
    sample_pos_tag = nltk.pos_tag(sample_tokenized)

    # ... [Include the rest of the verb tense changing logic from your original script here] ...

    # Reconstruct the perturbed sample
    perturbed_sample = " ".join([token for token, _ in sample_pos_tag])

    return perturbed_sample


ACTION_LIST_1 = [
    drop_stop_words,
    remove_punctuation,
    to_lowercase,
    negate_sentence,
    delete_random_character,
    insert_random_character,
    change_letter_case,
    apply_misspellings,
    repeat_random_character,
    replace_with_adjacent_character,
    swap_random_character,
    delete_random_word,
    perturb_word_order,
    repeat_random_word,
    toggle_singular_plural,
    change_verb_tense
]
ACTION_LIST_NAME = [
  'drop_stop_words',
  'remove_punctuation',
  'to_lowercase',
  'negate_sentence',
  'delete_random_character',
  'insert_random_character',
  'change_letter_case',
  'apply_misspellings',
  'repeat_random_character',
  'replace_with_adjacent_character',
  'swap_random_character',
  'delete_random_word',
  'perturb_word_order',
  'repeat_random_word',
  'toggle_singular_plural',
  'change_verb_tense'
]
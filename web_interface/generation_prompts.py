"""Shared validation for generation prompts and their independent visual checks."""

MAX_PROMPT_CHARS = 10000
MAX_PROMPT_CHECKS = 128
MAX_CHECK_QUESTION_CHARS = 4000
MAX_CHECK_ANSWER_CHARS = 1000
MAX_CHECK_SKILL_CHARS = 200


def validate_prompt(value):
    """Return the complete prompt; validation must not rewrite the requested scene."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('is empty or missing; choose rows with nonempty prompts')
    if len(value) > MAX_PROMPT_CHARS:
        raise ValueError(f'exceeds the {MAX_PROMPT_CHARS} character limit')
    if any(ord(char) < 32 and char not in '\t\n\r' for char in value):
        raise ValueError('contains unsupported control characters')
    return value


def _check_text(value, name, limit):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(ord(char) < 32 and char not in '\t\n\r' for char in value)):
        raise ValueError(f'{name} must be nonempty text of at most {limit} characters')
    return value.strip()


def normalize_prompt_checks(value):
    """Normalize pair or object checks without silently discarding malformed entries."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_PROMPT_CHECKS:
        raise ValueError(f'Prompt checks must be a list of at most {MAX_PROMPT_CHECKS} question/answer checks.')
    checks = []
    for index, entry in enumerate(value, 1):
        try:
            if isinstance(entry, dict):
                if not {'question', 'answer'} <= set(entry) or set(entry) - {'question', 'answer', 'skill'}:
                    raise ValueError('use question and answer fields, with an optional skill field')
                question, answer = entry['question'], entry['answer']
                skill = entry.get('skill')
            elif isinstance(entry, list) and len(entry) == 2:
                question, answer = entry
                skill = None
            else:
                raise ValueError('use a question/answer pair or an object with question and answer fields')
            check = dict(question=_check_text(question, 'question', MAX_CHECK_QUESTION_CHARS),
                         answer=_check_text(answer, 'answer', MAX_CHECK_ANSWER_CHARS))
            if skill is not None:
                check['skill'] = _check_text(skill, 'skill', MAX_CHECK_SKILL_CHARS)
            checks.append(check)
        except ValueError as exc:
            raise ValueError(f'Prompt check {index}: {exc}.') from exc
    return checks

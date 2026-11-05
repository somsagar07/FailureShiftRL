"""Evidence labels for measured outcomes; search rewards are not probabilities."""
import re


ANALYSIS_VERSION = 1
STATUSES = ('failure', 'candidate', 'no_failure_observed', 'inconclusive')


def note(code, message):
    return {'code': code, 'message': message}


def _unique(notes):
    return list({(item['code'], item['message']): item for item in notes}.values())


def assess_measurement(task, score, output, baseline, *, labelled=False):
    """Describe the evidence without upgrading automated flags to ground truth."""
    metrics = score['metrics']
    flagged = score['failure']
    status = 'candidate' if flagged else 'no_failure_observed'
    reasons, warnings = [], list(output.get('input_analysis', {}).get('warnings', []))
    basis = 'automated_review'
    if task == 'classification':
        basis = 'provided_label' if labelled else 'prediction_consistency'
        if flagged and labelled:
            status = 'failure'
            if metrics.get('baseline_error'):
                reasons.append(note('baseline_error', 'The prediction differs from the provided label; the original was also incorrect.'))
            else:
                reasons.append(note('new_label_error', 'The original matched the provided label; the edited input does not.'))
        elif flagged:
            reasons.append(note('prediction_changed', 'The edited prediction differs from the original prediction. A label is needed to identify an error.'))
        elif labelled:
            reasons.append(note('label_matched', 'The prediction matches the provided label.'))
        else:
            reasons.append(note('prediction_unchanged', 'The prediction did not change; correctness has not been established.'))
        if output.get('input_analysis', {}).get('status') == 'review':
            status = 'inconclusive'
    elif task == 'summarization':
        basis = 'reference_and_source_review' if 'source_support' in metrics or output.get('grounding') else 'reference_review'
        grounding = output.get('grounding', {})
        warnings.extend(grounding.get('warnings', []))
        if metrics.get('rouge_l', 1.) < .5:
            reasons.append(note('reference_overlap', 'Low reference overlap. A valid paraphrase can also have low overlap.'))
        if metrics.get('nonrepetition', 1.) < .7:
            reasons.append(note('repetition', 'The output repeats many word pairs.'))
        support = metrics.get('source_support')
        if isinstance(support, (int, float)) and support < .5:
            reasons.append(note('source_support', 'The NLI model found weak source support; inspect the claim evidence.'))
        for warning in output.get('summary_checks', {}).get('warnings', []):
            warnings.append(warning)
            if status == 'no_failure_observed':
                status = 'candidate'
        if grounding and not grounding.get('reliable', False):
            status = 'inconclusive'
        if baseline.get('grounding') and not baseline['grounding'].get('reliable', False):
            warnings.append(note('baseline_grounding_unreliable', 'The original output has an unstable or incomplete NLI assessment; the paired reward uses reference overlap only.'))
            status = 'inconclusive'
        if output.get('input_coverage', {}).get('truncated'):
            warnings.append(note('input_truncated', 'The model saw only part of the article. The complete reference may contain information outside that part.'))
            status = 'inconclusive'
        if baseline.get('input_coverage', {}).get('truncated'):
            warnings.append(note('baseline_input_truncated', 'The original input was truncated. Its output is not a complete-source baseline for this comparison.'))
            status = 'inconclusive'
        if output.get('input_analysis', {}).get('status') == 'review':
            status = 'inconclusive'
        if 'summary' in output and not output['summary'].strip():
            status = 'failure'
            reasons = [note('empty_output', 'The model returned an empty summary.')]
        elif not reasons:
            reasons.append(note('review_signal' if flagged else 'no_flag',
                                'The automatic review signal crossed its threshold.' if flagged else
                                'No automatic flag was raised. Factual correctness still requires review.'))
    elif task == 'generation':
        basis = 'visual_judge' if output.get('failure_scores') or 'checks_total' in output else 'similarity_review'
        components = output.get('failure_scores', {})
        threshold = metrics.get('failure_threshold', .5)
        for key, label in (('content', 'Scene content'), ('instructions', 'Presentation instructions'),
                           ('visual', 'Visual defects'), ('bias', 'Matched-group disparity')):
            value = components.get(key)
            if isinstance(value, (int, float)) and value > 0 and value >= threshold:
                reasons.append(note(key, f'{label} crossed the automatic review threshold.'))
        if components and components.get('bias') is None:
            warnings.append(note('bias_not_evaluated', 'Bias was not evaluated; it requires a configured matched-group comparison.'))
        if 'seed' in output:
            warnings.append(note('single_seed', 'This input/action was generated once. Replaying its cached result does not measure a failure probability.'))
        if not reasons:
            reasons.append(note('visual_review' if flagged else 'no_flag',
                                'The automatic image review flagged this output.' if flagged else
                                'No automatic flag was raised in the evaluated checks.'))
    else:
        reasons.append(note('review_signal', 'An evaluator supplied this review signal.'))
    return dict(version=ANALYSIS_VERSION, status=status, basis=basis,
                reasons=_unique(reasons), warnings=_unique(warnings))


def summarize_analysis(records, baseline, *, total_inputs, total_actions):
    """Count distinct measured pairs; DQN replay visits are never new evidence."""
    pairs = {(row['input_index'], row['action_id']): row for row in records}
    originals = {row['input_index']: row for row in baseline}
    def counts(rows):
        result = dict.fromkeys(STATUSES, 0)
        for row in rows:
            status = row.get('analysis', {}).get('status', 'inconclusive')
            result[status if status in result else 'inconclusive'] += 1
        return result
    return dict(version=ANALYSIS_VERSION, unique_pairs=len(pairs),
                inputs_evaluated=len({index for index, _ in pairs}), total_inputs=total_inputs,
                total_pairs=total_inputs*total_actions, counts=counts(pairs.values()),
                baseline_counts=counts(originals.values()), baseline_inputs=len(originals))


_NUMBER = re.compile(r'(?<!\w)\d+(?:[.,]\d+)*(?:\s*%|\b)')
_NEGATION = re.compile(r"\b(?:no|not|never|without|neither|nor)\b|\b\w+n['’]t\b|\bcannot\b", re.I)


def numeric_literals(text):
    """Literal evidence only: this does not prove numeric entailment or units."""
    return {re.sub(r'\s+', '', value) for value in _NUMBER.findall(text)}


def summary_checks(source, summary):
    introduced = sorted(numeric_literals(summary) - numeric_literals(source))
    warnings = []
    if introduced:
        warnings.append(note('new_number', 'Numbers absent from the evaluated source: ' + ', '.join(introduced[:20]) + '. They may be inferred or reformatted; verify them.'))
    return dict(new_numeric_literals=introduced[:20], warnings=warnings)


def input_analysis(task, original, edited, action):
    """Surface action assumptions and concrete changed numeric/negation tokens."""
    from core import action_semantic_risks
    if task == 'summarization' and original == edited:
        return dict(status='unchanged', warnings=[])
    # Historical reports and injected evaluators may have only an action ID.
    # Missing review metadata must not discard a completed model measurement.
    warnings = []
    if task == 'summarization' and not (action or {}).get('name'):
        risks = []
        warnings.append(note('action_metadata_unavailable',
            'This saved action has no current validity guidance. Review its expected target.'))
    elif not action:
        risks = []
    else:
        try:
            risks = action_semantic_risks(task, action)
        except ValueError:
            return dict(status='review', warnings=[note('action_metadata_unavailable',
                'This saved action has no current validity guidance. Review its expected target.')])
    warnings.extend(note('action_' + risk['semantic_risk'], risk['label'] + ': ' + risk['reason']) for risk in risks)
    if task == 'summarization':
        before, after = numeric_literals(original), numeric_literals(edited)
        if before != after:
            warnings.append(note('numbers_changed', 'The edit changes numeric literals. The original reference may no longer describe the edited input.'))
        # Expanded/contracted forms count alike. This catches removal/addition,
        # not changes in scope; the absence of a warning is not semantic proof.
        if len(_NEGATION.findall(original)) != len(_NEGATION.findall(edited)):
            warnings.append(note('negation_changed', 'The edit changes negation markers. Review whether the expected answer should change.'))
    return dict(status='review' if warnings else 'no_risk_detected', warnings=_unique(warnings))

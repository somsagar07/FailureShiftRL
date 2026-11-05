"""Read completed local training artifacts without importing a model runtime."""
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


def load_starter_catalog(directory):
    from robotics import normalize_config
    root=Path(directory).resolve(); manifest=root/'catalog.json'
    result={'policies':[],'warnings':[]}
    if not manifest.exists(): return result
    try:
        if manifest.stat().st_size>2*1024**2: raise ValueError('Starter catalog exceeds 2 MiB.')
        data=json.loads(manifest.read_text())
        entries=data.get('policies') if isinstance(data,dict) else None
        if not isinstance(entries,list) or len(entries)>100: raise ValueError('Starter catalog must contain up to 100 policies.')
    except (OSError,ValueError) as exc:
        result['warnings'].append(f'Could not read starter policies: {exc}')
        return result
    seen=set()
    for entry in entries:
        identifier=entry.get('id','unknown') if isinstance(entry,dict) else 'unknown'
        try:
            if not isinstance(entry,dict): raise ValueError('Invalid policy entry.')
            if not isinstance(identifier,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',identifier) or identifier in seen:
                raise ValueError('Each starter policy needs a distinct ID.')
            policy=entry.get('policy',{})
            if policy.get('format')!='torchscript' or policy.get('source')!='local':
                raise ValueError('Starter policies must be exported local TorchScript checkpoints.')
            relative=policy.get('name','')
            if not isinstance(relative,str) or not relative or '\\' in relative or any(
                    part in ('','..','.') for part in relative.split('/')) or PurePosixPath(relative).is_absolute():
                raise ValueError('Checkpoint must be a relative path inside starter-policies.')
            checkpoint=(root/relative).resolve()
            if not checkpoint.is_relative_to(root): raise ValueError('Checkpoint resolves outside starter-policies.')
            if not checkpoint.is_file() or checkpoint.stat().st_size==0: raise ValueError('Checkpoint is not ready.')
            expected=entry.get('checkpoint_sha256')
            if not isinstance(expected,str) or not re.fullmatch(r'[0-9a-f]{64}',expected):
                raise ValueError('The evaluated checkpoint digest is missing.')
            with checkpoint.open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:
                raise ValueError('The checkpoint has changed since these results were measured; finish training and refresh.')
            candidate=copy.deepcopy(entry); candidate['policy']=dict(policy,name=str(checkpoint))
            normalized=normalize_config(candidate)
            label=candidate.get('label',identifier)
            if not isinstance(label,str) or not label.strip() or len(label)>160: raise ValueError('Invalid starter label.')
            evaluation=candidate.get('evaluation',{})
            count=evaluation.get('episodes'); successes=evaluation.get('successes')
            if type(count) is not int or count<1 or type(successes) is not int or not 0<=successes<=count:
                raise ValueError('Completed held-out success and episode counts are required.')
            rate=successes/count; supplied=evaluation.get('success_rate',rate)
            if type(supplied) not in (int,float) or not math.isfinite(supplied) or abs(rate-supplied)>.005:
                raise ValueError('Success rate differs from the measured episode counts.')
            candidate.update(label=label,environment=normalized['environment'],mapping=normalized['mapping'],
                instruction=normalized['instruction'],policy=normalized['policy'])
            candidate['evaluation']['success_rate']=rate
            json.dumps(candidate,allow_nan=False)
            result['policies'].append(candidate); seen.add(identifier)
        except (ValueError,TypeError,AttributeError,OSError) as exc:
            result['warnings'].append(f'Starter policy {identifier} is unavailable: {exc}')
    return result

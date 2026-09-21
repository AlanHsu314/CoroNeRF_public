#############################################
## NAMING RUNS
#############################################


from __future__ import annotations

from typing import Union, Any
import hashlib
import json


def short_hash(data: dict, n: int = 8) -> str:
    '''
    small hash function for data
    '''
    payload = json.dumps(data, sort_keys = True, separators=(',', ':'))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:n]

def _compact_scalar(value) -> str:
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)

def sanitize_value(value) -> str:
    '''
    compact, path-safe string for benchmark run names
    '''
    if isinstance(value, (list, tuple)):
        digest = short_hash({'value': list(value)}, n=6)
        return f'list-{digest}'

    if isinstance(value, dict):
        digest = short_hash({'value': value}, n=6)
        return f'dict-{digest}'

    text = str(value)
    text = text.replace('/', '-')
    text = text.replace('\\', '-')
    text = text.replace(' ', '')
    text = text.replace("'", '')
    text = text.replace('"', '')
    return text

def shorten_key(key: str) -> str:
    leaf = key.split('.')[-1]
    mapping = {
        'hidden_dim': 'hdim',
        'num_hidden': 'layers',
        'lambda_smooth': 'lsmooth',
        'renderer_model': 'rend',
        'occ_res': 'occ',
        'alpha_by_global_channel': 'alphaAll',
        'alpha': 'alpha',
    }
    return mapping.get(leaf, leaf)

def make_run_name(
    experiment_name: str,
    name_overrides: dict,
    seed: Union[int, None] = None,
)-> str:
    '''
    creates a (somewhat) readable and stable name
    '''
    parts = [experiment_name]

    # keep readable swept params in the folder name
    # keys = sorted(flat_overrides.keys())
    # for key in keys[:max_parts]:
    #     short_key = key.split('.')[-1]
    #     parts.append(f'{short_key}-{sanitize_value(flat_overrides[key])}')

    for key in sorted(name_overrides.keys()):
        parts.append(f'{shorten_key(key)}-{sanitize_value(name_overrides[key])}')

    if seed is not None:
        parts.append(f'seed-{seed}')

    # add hash
    digest = short_hash({'experiment_name': experiment_name, 
                         'name_overrides': name_overrides,
                         'seed': seed})
    parts.append(digest)

    # join full name and return
    name = '__'.join(parts)

    # safeguard name of run
    if len(name) > 100:
        name = f'{experiment_name}__seed-{seed}__{digest}' if seed is not None else f'{experiment_name}__{digest}'

    return name






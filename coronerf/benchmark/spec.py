#############################################################
### loads spec yaml
#############################################################

from __future__ import annotations
from typing import Union
from pathlib import Path
from .config import load_yaml

def load_benchmark_spec(path: Union[Path, str]) -> dict:
    '''
    load bench mark spec, specifiying the benchmark study and the list of experiments
    '''
    # resolve relative path of benchmark spec
    spec_path = Path(path).resolve()
    spec = load_yaml(spec_path)

    if 'benchmark' not in spec:
        raise ValueError("Benchmark spec must contain top-level 'benchmark'")
    
    bench = spec['benchmark']

    # check for required keys in benchmark
    required = ['name', 'base_config', 'output_dir', 'experiments']
    for key in required:
        if key not in bench:
            raise ValueError(f'Benchmark spec missing required key: benchmark.{key}')
    
    # resolve base_config relative to spec file location
    base_config = Path(bench['base_config'])
    if not base_config.is_absolute():
        base_config = (spec_path.parent / base_config).resolve()
    bench['base_config'] = str(base_config)

    # resolve output_dir relative to spec file location
    output_dir = Path(bench['output_dir'])
    if not output_dir.is_absolute():
        output_dir = (spec_path.parent / output_dir).resolve()
    bench['output_dir'] = str(output_dir)

    # experiments must be a non-empty list
    if not isinstance(bench['experiments'], list) or len(bench['experiments']) == 0:
        raise ValueError('benchmark.experiments must be a non-empty list')

    # add some defaults
    bench.setdefault('fixed', {})
    bench.setdefault('seeds', [0xC0FFEE])

    # check each experiment
    for exp in bench['experiments']:
        if 'name' not in exp:
            raise ValueError("Each experiment must have a 'name'")
        exp.setdefault('overrides', {})
        exp.setdefault('grid', {})

    # finally add some metadata
    bench['_spech_path'] = str(spec_path)
    bench['_spec_dir'] = str(spec_path.parent)

    # return pre-processed spec
    return spec








#############################################
## expands benchmark spec into concrete runs
#############################################


from __future__ import annotations
from typing import Union

from itertools import product

from .naming import make_run_name


def cartesian_product_dict(grid: dict[str, list]) -> list[dict]:
    '''
    [INPUT]
    dictionary of N lists, each defining an axis of the grid

    [OUTPUTS]
    list of dictionaries, each pertaining to a run corresponding to a point on the N-dim grid
    '''

    if not grid:
        return [{}]
    
    # get grid axes
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    # create list of runs
    rows = []
    for combo in product(*values):
        rows.append({k: v for k, v in zip(keys, combo)}) # create dict for a run
    
    return rows

def expand_benchmark_spec(spec: dict) -> list[dict]:
    '''
    created an expanded benchmark spec with specs for each run
    '''

    bench = spec['benchmark']
    fixed = bench.get('fixed', {})
    seeds = bench.get('seeds', [0xC0FFEE])

    expanded = []

    # loop through experiment specs
    for exp in bench['experiments']:
        exp_name = exp['name']
        exp_overrides = exp.get('overrides', {})
        exp_grid = exp.get('grid', {})

        # expand the exp_grid
        grid_rows = cartesian_product_dict(exp_grid)
        sweep_keys = sorted(exp_grid.keys())

        # begin to create a run spec
        for row in grid_rows:
            # start to build merged dict
            merged = {}
            merged.update(fixed)
            merged.update(exp_overrides)
            merged.update(row)

            # per seed
            for seed in seeds:
                merged_with_seed = dict(merged)
                merged_with_seed['seed'] = seed

                # make the run
                run_name = make_run_name(
                    experiment_name = exp_name,
                    name_overrides = row,        # just hash the grid-combo values
                    seed = seed,
                )

                # add to list of runs
                expanded.append({
                    'benchmark_name': bench['name'],
                    'experiment_name': exp_name,
                    'seed': seed,
                    'run_name': run_name,
                    'overrides': merged_with_seed,
                    'sweep_keys': sweep_keys,
                    'sweep_values': dict(row),
                })

    # flattended list of run specs
    return expanded
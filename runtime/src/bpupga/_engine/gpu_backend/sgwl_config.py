from dataclasses import dataclass
from baseline.author_adapters import AuthorSGWLConfig

@dataclass
class DegreePriorConfig(AuthorSGWLConfig):
    probability_shift: float=1.
    probability_power: float=.1
    node_prior: float=10.
    node_prior_norm: str='absolute'
    squared_cost: str='elementwise'
    beta: float=.001

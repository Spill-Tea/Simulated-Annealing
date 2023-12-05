# Simulated-Annealing

Simple and Clean Interface to perform Simulated Annealing on NP Hard Problems.


### Table of Contents
- [Simulated-Annealing](#simulated-annealing)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Temperature Cooling Paradigms](#temperature-cooling-paradigms)
    - [Examples](#examples)
      - [_Traveling Salesman Problem (TSP)_](#traveling-salesman-problem-tsp)
    - [License](#license)


### Installation
```bash
pip install git+https://github.com/Spill-Tea/Simulated-Annealing@main
```

### Temperature Cooling Paradigms

Below, we have an image of all cooling paradigms available in the package, over 1000 steps and alpha = 0.85, in a temperature range of 0 to 100. Linear cooling is not truly useful in practice, but is helpful to compare between more relevant cooling methods.
![Cooling](docs/cooling_paradigms.png)


### Examples
#### _Traveling Salesman Problem (TSP)_
```python
import numpy as np
from Annealing import anneal
from Annealing import cooling
from Annealing import fitness

# Define Interface for TSP
class TSP(anneal.AnnealingBase):
    def mixing(self, index, nshuffle):
        super().mixing(index, nshuffle)

    def subsample(self, indices: np.ndarray) -> np.ndarray:
        return super().subsample(indices)

# Random XYZ Coordinates
seed = np.random.default_rng(23)
coordinates = seed.uniform(0., 100., (50, 3))

# Instantiate TSP Interface
tsp = TSP(
    coordinates,
    cooling.InverseCooling(1000, 0.9),
    fitness.CircularEuclidean()
)
intial = tsp.fitness(coordinates)  # 3571.1151820333043
print(f"Initial Distance: {initial}")

# You might want to simualte several times
result = travel.simulate(nswaps=2)
print(f"Proposed Minimum Distance: {travel.best}")

# Best Observed Performance: 1088.2251082017222
best_idx = np.asarray([ 
     0,  6, 13, 28, 37,  3, 19, 44, 34, 45,
     2, 49, 30, 41, 25, 31, 18, 38,  8, 29, 
    24, 14, 27, 20, 33, 10, 42, 48, 40,  7,
     5, 26, 46, 12,  9, 36,  1, 32, 47, 22,
    39, 16,  4, 43, 23, 17, 35, 15, 11, 21,
])

print(f"Best Observed Minimum Distance: {tsp.fitness(coordinates[best_idx])}")
```
![Cooling](docs/TSP_best.png)


### License
![MIT License](LICENSE)

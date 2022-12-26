# ![logo](https://raw.githubusercontent.com/neka-nat/fastmunk/master/assets/logo.png)

Python package for fast munkres algorithm.

# Installation

```sh
pip install fastmunk
```

## Getting started

```py
import fastmunk

mat = np.array([
    [5, 9, 1],
    [10, 3, 2],
    [8, 7, 4],
], dtype=np.float64)

m = fastmunk.FastMunk()
m.compute(mat)
```

## Benchmark

```sh
cd examples
python benchmark.py
```

Output

```sh
FastMunk:  0.02310633659362793 [s]
Munkres:  0.5914878845214844 [s]
```

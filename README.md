# MoLeSq

Pronounced like "mollusc".

A python implementation of the Moving Least Squares algorithm
for transforming sets of points from one space to another,
as published in [Schaefer et al. (2006)][1].

Repackaged from [implementation by Casey Schneider-Mizell][2].

[1]: https://doi.org/10.1145/1179352.1141920
[2]: https://github.com/ceesem/catalysis/blob/master/catalysis/transform.py

## Usage

Control points and points of interest are given as numpy arrays.

```python
from molesq import Transformer

tran = Transformer(my_control_points, my_deformed_control_points)
deformed = tran.transform(my_points_of_interest)

undeformed = tran.transform(deformed, reverse=True)

```

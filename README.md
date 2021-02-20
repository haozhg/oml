# osysid
Efficient online adaptive linear/nonlinear system identification and control.
```
pip install osysid
```

## Highlights
- Efficiently online adaptive linear/nonlinear model learning. Any nonlinear and/or time-varying system is locally linear, as long as the model is updated in real-time wrt to measurement.
- Optimal in terms of both time and space complexity. 
- The time complexity (multiply–add operation for one iteration) is O(n^2)
- Space complexity is O(n^2), where n is the problem dimension. 
- This local model can be used for short-horizon prediction 
and real-time control.
- A weighting factor (in (0, 1]) can be used to place more weight on recent data, thus making the model more adaptive.

## Online system identification algorithm description
For more details, see this [paper](https://epubs.siam.org/doi/pdf/10.1137/18M1192329).

### Unknown dynamical system
Suppose we have a (discrete) nonlinear and time-varying system 
- x(k+1) = f(t(k), x(k), u(k))
- y(k) = g(t(k), x(k), u(k))

and we have measurements x(k), u(k), y(k) for k = 0,1,...T

### Online linear model learning
We would like to learn an adaptive linear model
- x(k+1) = A*x(k) + B*u(k)
- y(k) = C*x(k) + D*u(k)

that fits the observation optimally. Based on Taylor expansion, any nonlinear/time-varying system is linear locally. Also, there are many tools for linear control, e.g, LQR, Kalman filter. However, we need to update this linear model efficiently in real-time whenever new measurement becomes available.

The problem can be formulated as an optimization problem, and at each time step k we need to solve a related but slightly different optimization problem. The optimal algorithm is achived through efficient reformulation of the problem. This package implements this algorithm, and for more detail pls refer to this [paper](https://epubs.siam.org/doi/pdf/10.1137/18M1192329).


### Online nonlinear model learning
In case that you want to fit a nonlinear model to the observed data. However, notice that it is not straightforward to justify this choice， since linear adaptive model is good enough as long as it is updated in real-time.

Suppose we want to fit a nonlinear model of this form
- x(k+1) = F * phi(x(k), u(k))
- y(k) = G * psi(x(k), u(k))

where phi(~, ~) and psi(~, ~) are known nonlinear nonlinear functions (e.g, linear or quadratic), F and G are unknown matrices of proper size. We aim to learn F and G from data.

The problem can also be formulated as the same optimization problem, and the same efficient algorithm works in this case.

## Use
### install
pip
```
pip install osysid
```

Manual install
```
git clone https://github.com/haozhg/osysid.git
cd osysid/
pip install -e .
```

### Tests
```
cd osysid/
python -m pytest .
```

### Demo
See `./demo` for python notebook to demo the algorithm for data-driven real-time closed loop control.

## Authors:
Hao Zhang 

## Reference
If you you used these algorithms or this python package in your work, please consider citing

```
Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta. "Online dynamic mode decomposition for time-varying systems." SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.
```

BibTeX
```
@article{zhang2019online,
  title={Online dynamic mode decomposition for time-varying systems},
  author={Zhang, Hao and Rowley, Clarence W and Deem, Eric A and Cattafesta, Louis N},
  journal={SIAM Journal on Applied Dynamical Systems},
  volume={18},
  number={3},
  pages={1586--1609},
  year={2019},
  publisher={SIAM}
}
```

## Date created
April 2017

## License
If you want to use this package, but find license permission an issue, pls contact me at `haozhang at alumni dot princeton dot edu`.

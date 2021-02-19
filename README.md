# osysid
Python/Matlab implementation of online system identification

## Highlights
- Efficiently online linear model learning
- Optimal in terms of both time and space complexity

## Online system identification algorithm description
### Online linear model learning


### Online Nonlinear model learning
Suppose that we have a nonlinear and time-varying system z(k) =
f(t(k-1),z(k-1),u(k-1)). We aim to build local linear model in the form
of z(k) = F(z(k),t(k)) z(k-1) + G(z(k),t(k)) u(k-1), where F(z(k),t(k)), 
G(z(k),t(k)) are two matrices. We define x(k) = [z(k-1); u(k-1)], y(k) =
z(k), and A(k) = [F(z(k),t(k)), G(z(k),t(k))]. Then the local linear 
model can be written as y(k) = A(k) x(k).  
We can also build nonlinear model by defining nonlinear observable x(k)
of state z(k-1) and control u(k-1). For example, for a nonlinear system 
z(k)=z(k-1)^2+u(k-1)^2, we can define x(k) = [z(k-1)^2;u(k-1)^2], y(k) =
z(k), then it can be written in linear form y(k) = A*x(k), where A=[1,1].  
At time step k, we assume that we have access to z(j),u(j),j=0,1,2,...k.
Then we have access to x(j), y(j), j=1,1,2,...k. We define two matrices
X(k) = [x(1),x(2),...,x(k)], Y(k) = [y(1),y(2),...,y(k)], that contain 
all the past snapshot. The best fit to the data is Ak = Yk*pinv(Xk).  
An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more 
weight on recent data can be incorporated into the definition of X(k) and
Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)*x(2),...,
sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,
sigma^(1)*y(k-1),y(k)].  
At time step k+1, we need to include new snapshot pair x(k+1), y(k+1).
We would like to update the general DMD matrix Ak = Yk*pinv(Xk) recursively 
by efficient rank-1 updating online DMD algorithm.  
Therefore, A(k) explains the most recent data and is considered to be 
the local linear model for the original nonlinear and/or time-varying 
system. This local linear model can be used for short-horizon prediction 
and real-time control.  
The time complexity (multiply–add operation for one iteration) is O(n^2), and space complexity is O(n^2), where n is the problem dimension.  

## Installation
### pip
```
pip install osysid
```

### Manual install
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

# osysid
Python/Matlab implementation of online system identification

## Online system identification algorithm description
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
Download online system identification implementation from github
`git clone https://github.com/haozhg/osysid.git`

## Implementation
1.**OnlineSysId.m** implements **OnlineSysId** class in Matlab.   
3.**onlinesysid.py** implements **OnlineSysId** class in Python.  

## Documentation
Matlab:  
type **help OnlineSysId** for **OnlineSysId** class documentation.  
Python:  
type **help(onlinesysid.OnlineSysId)** for **OnlineSysId** class documentation.  

## Demos
1.**OnlineSysId_demo.m** demos the use of Matlab **OnlineSysId** class.  
3.**onlinesysid_demo.py** demos the use of Python **OnlineSysId** class.  

## Authors:
Hao Zhang  
Clarence W. Rowley

## Reference:
Hao Zhang, Clarence W. Rowley,
``Real-time control of nonlinear and time-varying systems based on 
online linear system identification", in production, 2017.

## Date created:
June 2017


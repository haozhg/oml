% OnlineSysID is a class that implements the online system identification.
% The time complexity (multiply-add operation for one iteration) is O(4n^2)
% , and space complexity is O(2n^2), where n is the problem dimension.
% It works for both online linear and nonlinear system identification.
% 
% Algorithm description:
% Suppose that we have a nonlinear and time-varying system z(k) =
% f(t(k-1),z(k-1),u(k-1)). We aim to build local linear model in the form
% of z(k) = F(z(k),t(k)) z(k-1) + G(z(k),t(k)) u(k-1), where F(z(k),t(k)), 
% G(z(k),t(k)) are two matrices. We define x(k) = [z(k-1); u(k-1)], y(k) =
% z(k), and A(k) = [F(z(k),t(k)), G(z(k),t(k))]. Then the local linear 
% model can be written as y(k) = A(k) x(k).
% We can also build nonlinear model by defining nonlinear observable x(k)
% of state z(k-1) and control u(k-1). For example, for a nonlinear system 
% z(k)=z(k-1)^2+u(k-1)^2, we can define x(k) = [z(k-1)^2;u(k-1)^2], y(k) =
% z(k), then it can be written in linear form y(k) = A*x(k), where A=[1,1].
% At time step k, we assume that we have access to z(j),u(j),j=0,1,2,...k.
% Then we have access to x(j), y(j), j=1,1,2,...k. We define two matrices
% X(k) = [x(1),x(2),...,x(k)], Y(k) = [y(1),y(2),...,y(k)], that contain 
% all the past snapshot. The best fit to the data is Ak = Yk*pinv(Xk).
% An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more 
% weight on recent data can be incorporated into the definition of X(k) and
% Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)*x(2),...,
% sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,
% sigma^(1)*y(k-1),y(k)].
% At time step k+1, we need to include new snapshot pair x(k+1), y(k+1).
% We would like to update the general DMD matrix Ak = Yk*pinv(Xk) recursively 
% by efficient rank-1 updating online DMD algorithm.
% Therefore, A(k) explains the most recent data and is considered to be 
% the local linear model for the original nonlinar and/or time-varying 
% system. This local linear model can be used for short-horizon prediction 
% and real-time control.
%
% Usage:
% osysid = OnlineSysId(n,q,weighting)
% osysid.initialize(X0,Y0)
% osysid.initilizeghost()
% osysid.update(x,y)
%
% properties:
% n: state dimension
% q: observable vector dimension, state dimension + control dimension for
% linear system identification case
% weighting: weighting factor in (0,1]
% timestep: number of snapshot pairs processed
% A: general DMD matrix, size n by q
% P: matrix that contains information about past snapshots, size q by q
%
% methods:
% initialize(X0, Y0), initialize online DMD algorithm with k0 snapshot
%                     pairs stored in (X0, Y0)
% initializeghost(),  initialize online DMD algorithm with epsilon 
%                     small (1e-15) ghost snapshot pairs before t=0
% update(x,y), update when new snapshot pair (x,y) becomes available
%
% Authors: 
% Hao Zhang
% Clarence W. Rowley
% 
% Reference:
% Hao Zhang, Clarence W. Rowley,
% ``Real-time control of nonlinear and time-varying systems based on 
% online linear system identification", in production, 2017.
% 
% Created:
% June 2017.
%
% To look up the documentation in the command window, type help OnlineSysId


classdef OnlineSysId < handle
    properties
        n = 0;                      % state dimension
        q = 0;                      % observable dimension
        weighting = 1;                 % weighting factor in (0,1]
        timestep = 0;               % number of snapshots processed
        A;          % y(k) = Ax(k), size n by q
        P;          % matrix that contains information about past snapshots
    end
    
    methods
        function obj = OnlineSysId(n,q,weighting)
            % Creat an object for OnlineSysId
            % Usage: osysid = OnlineSysId(n,q,weighting)
            if nargin == 3
                obj.n = n;
                obj.q = q;
                obj.weighting = weighting;
                obj.A = zeros(n,q);
                obj.P = zeros(q,q);
            end
        end
        
        function initialize(obj, X0, Y0)
            % Initialize OnlineSysId with k0 snapshot pairs stored in (X0, Y0)
            % Usage: osysid.initialize(X0,Y0)
            k0 = length(X0(1,:));
            if(obj.timestep == 0 && rank(X0) == obj.q)
                weight = (sqrt(obj.weighting)).^(k0-1:-1:0);
                X0 = X0.*weight;
                Y0 = Y0.*weight;
                obj.A = Y0*pinv(X0);
                obj.P = inv(X0*X0')/obj.weighting;
            end
            obj.timestep = obj.timestep + k0;
        end
        
        function initializeghost(obj)
            % Initialize OnlineSysId with epsilon small (1e-15) ghost 
            % snapshot pairs before t=0
            % Usage: osysid.initilizeghost()
            epsilon = 1e-15;
            obj.A = zeros(obj.n, obj.q);
            obj.P = (1/epsilon)*eye(obj.q);
        end
        
        function update(obj, x, y)
            % Update the online DMD computation with a new pair of snapshots (x,y)
            % Here, if the (discrete-time) dynamics are given by z(k) = 
            % f(t(k-1),z(k-1),u(k-1)), then x=[z(k-1);u(k-1)], y=z(k).
            % Usage: osysid.update(x, y)
            
            % compute P*x matrix vector product beforehand
            Px = obj.P*x;
            % Compute gamma
            gamma = 1/(1+x'*Px);
            % Update A
            obj.A = obj.A + (gamma*(y-obj.A*x))*Px';
            % Update P, group Px*Px' to ensure positive definite
            obj.P = (obj.P - gamma*(Px*Px'))/obj.weighting;
            % ensure P is SPD by taking its symmetric part
            obj.P = (obj.P+(obj.P)')/2;
            % time step + 1
            obj.timestep = obj.timestep + 1;
        end
    end
end
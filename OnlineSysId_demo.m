% An example to demonstrate online linear system identification
% 
% We demonstrate the use of OnlineSysId class with a simple linear system.
% Take a 2D time-varying system dz/dt=A(t)z(t)+B(t)u(t), where A(t) and B(t)
% are slowly varying with time. In particular, we take A(t)=(1+eps*t)*A0,
% B(t)=(1+eps*t)*B0, and eps = 0.1 is small. It is discretize with
% time step dt = 0.1. Denote the discrete system as z(k)=A(k)z(k-1)+
% B(k)u(k-1). We define x(k) = [z(k-1);u(k-1)], y(k) = z(k),
% F(k)=[A(k),B(k)], then the original system can be written as y(k)=F(k)x(k).
% 
% At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)], 
% Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs.
% The best fit to the data is Fk = Yk*pinv(Xk).
% 
% An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more 
% weight on recent data can be incorporated into the definition of X(k) and
% Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)t*x(2),?,
% sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,
% sigma^(1)*y(k-1),y(k)].
% 
% At time step k+1, we need to include new snapshot pair x(k+1), y(k+1).
% We would like to update the DMD matrix Fk = Yk*pinv(Xk) recursively 
% by efficient rank-1 updating online DMD algorithm.
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

% define dynamics
A0 = [0,1;-1,-0.1]; B0 = [0;1]; epsilon = 1e-1;
dyn = @(t,z,u) (1+epsilon*t)*A0*z + (1+epsilon*t)*B0*u;
% set up simulation parameters
dt = 1e-1;
tmax = 10; tc = 0.5;
kmax = floor(tmax/dt); kc = floor(tc/dt);
tspan = 0:dt:tmax;
% control input sine wave
gamma = 1; omega = 5;
% dimensions
n = 2; p = 1; q = n + p;
% online system identification set up
weighting = 0.01^(2/kc);
osysid = OnlineSysId(n,q,weighting);
osysid.initializeghost();
% store data matrices
x = zeros(q,kmax); y = zeros(n,kmax);
Aerror = zeros(kmax,1); Berror = zeros(kmax,1);


% initial condition,state and control
z0 = [1;0]; u0 = 0;
zk = z0; uk = u0;
% system simulation
for k=1:kmax
    % update state x(k) = [z(k-1);u(k-1)]
    x(:,k) = [zk;uk];
    % forward the system for one step
    zk = zk + dt*dyn((k-1)*dt,zk,uk);
    % update control input according to sine wave
    uk = gamma*sin(omega*k*dt);
    % update state y(k) = z(k)
    y(:,k) = zk;
    % use new data to update online system identification
    osysid.update(x(:,k),y(:,k));
    % model error at time k
    Ak = eye(n)+dt*(1+epsilon*k*dt)*A0;
    Bk = dt*(1+epsilon*k*dt)*B0;
    Aerror(k) = norm(osysid.A(:,1:n)-Ak,'fro')/norm(Ak,'fro');
    Berror(k) = norm(osysid.A(:,n+1:q)-Bk,'fro')/norm(Bk,'fro');
end


% visualize snapshots and control inputs
figure, hold on
plot(tspan(1:kmax),y(1,:),'x-',tspan(1:kmax),y(2,:),'o-',tspan(1:kmax),x(3,:),'s-','LineWidth',2)
xlabel('Time','Interpreter','latex')
title('Snapshots','Interpreter','latex')
fl = legend('$z_1(t)$','$z_2(t)$','$u(t)$');
set(fl,'Interpreter','latex','Location','northwest','Box','off');
xlim([0,tmax]), ylim([-2,2])
box on
set(gca,'FontSize',20,'LineWidth',2)

% visualize model error
figure, hold on
plot(tspan(1:kmax),Aerror,'x-',tspan(1:kmax),Berror,'o-','LineWidth',2)
xlabel('Time','Interpreter','latex')
title('Online DMD model error','Interpreter','latex')
fl = legend('$\frac{||A_{DMD}(t)-A(t)||_F}{||A(t)||_F}$','$\frac{||B_{DMD}(t)-B(t)||_F}{||B(t)||_F}$');
set(fl,'Interpreter','latex','Location','northeast','Box','off');
xlim([0,tmax]), ylim([0,0.2])
box on
set(gca,'FontSize',20,'LineWidth',2)
function [ X,gamma,gamma_d ]=DGP2(T,p,pp,h)
% 生成了T个真正符合VAR(1)-GARCH(1,1)的样本,gamma和gamma_d是13*T维的
% cf JoE_Gao，初始值是由非时变的VAR(1)-GARCH(1,1)生成的
% VAR(1)-GARCH(1,1): x_t=A1(τt)x_{t-1}+vt;
%vt = diag(ht)^(1/2)ηt, ηt = Ω^1/2(τt)εt
%ht = c0(τt)+C1(τt)(v_{t-1}.v_{t-1})+D1(τt)h_{t-1}
% p对应的是x的滞后项，pp对应的是v的滞后项
rng(0);
Tdiscard = 200; % 非时变的VARMA的样本数量为200个
m=2;
A1 = [0.5 * exp(-0.5), -0.2 * exp(-1);
      -0.2,  0.6 * exp(-0.5)];
c0 = [2 + exp(-0.5), 3 + 0.2]';
C1 = [0.4 + 0.05, 0.05 * (-0.5)^2 + 0.05;    
      0.05 * (-0.5)^2 + 0.05,  0.4];
D1 = [0.3, 0;    
      0,  0.3];
      
Omega = [1,  0;
         0,  1];
omega = chol(Omega);
    
epsilon_discard = mvnrnd(zeros(m,1),eye(m),Tdiscard);
eta_discard = epsilon_discard * omega;
    
Vdiscard = zeros(Tdiscard,m);
H_discard = zeros(Tdiscard,m);
    
H_discard(1,:) = [1,1];
Vdiscard(1:pp,:) =  eta_discard(1:pp,:) * mpower(diag(H_discard(1,:)),0.5);
    
for t = 1:(Tdiscard-pp)
    H_discard(pp+t,:) = c0' + (Vdiscard(pp+t-1,:).*Vdiscard(pp+t-1,:))*(C1)'+ H_discard(pp+t-1,:)*D1';   % 这里H_discard(pp+t-qq,:),这里其实默认qq=1；
    Vdiscard(pp+t,:) = eta_discard(pp+t,:) * mpower(diag(H_discard(pp+t,:)),0.5);
end
    
Xdiscard = zeros(Tdiscard,m);
Xdiscard(1,:) = Vdiscard(1,:);
for t = 2:Tdiscard
    Xdiscard(t,:) = Xdiscard(t-1,:)*A1' + Vdiscard(t,:);
end
    
% plot([1:Tdiscard],Vdiscard(:,1),'k',[1:Tdiscard],Vdiscard(:,2),'r');
% plot([1:Tdiscard],H_discard(:,1),'k',[1:Tdiscard],H_discard(:,2),'r');
% plot([1:Tdiscard],Xdiscard(:,1),'k',[1:Tdiscard],Xdiscard(:,2),'r');
        
V = [Vdiscard(end,:);zeros(T+h,m)];  % vt需要滞后一阶
epsilon = mvnrnd(zeros(m,1),eye(m),T+h);
eta = [eta_discard(end,:);zeros(T+h,m)]; % eta_t需要滞后一阶
H_true = [H_discard(end,:);zeros(T+h,m)]; % H_t需要滞后一阶，这个个数其实是由p,pp,qq决定的
    
c = zeros(m,T+h);    % the time-varying coefficients
c_d = zeros(m,T+h);  % the first-order derivative of time-varying coefficients
C = zeros(m,m,T+h);
C_d = zeros(m,m,T+h);
D = zeros(m,m,T+h);
D_d = zeros(m,m,T+h);
Omega = zeros(m,m,T+h);
% 先生成V：-pp:1:T+h
% 这里所有系数向量都多了一个_d是干嘛的？ 是因为用局部线性来估计，
for t = 1:T+h
    c(:,t) = [2+exp(0.5*t/(T+h) - 0.5);3 + 0.2*cos(t/(T+h))];
    c_d(:,t) = [0.5*exp(0.5*t/(T+h) - 0.5); -0.2*sin(t/(T+h))];
    C(:,:,t) = [0.4 + 0.05*cos(t/(T+h)), 0.05*(t/(T+h) - 0.5)^2+0.05;
               0.05*(t/(T+h) - 0.5)^2+0.05, 0.4 + 0.05*sin(t/(T+h))];
    C_d(:,:,t) = [-0.05*sin(t/(T+h)),0.1*(t/(T+h)-0.5);
                  0.1*(t/(T+h)-0.5), 0.05*cos(t/(T+h))];
    D(:,:,t) = [0.4 - 0.1*cos(t/(T+h)), 0;
                0, 0.3 - 0.1*sin(t/(T+h))];
    D_d(:,:,t) = [0.1*sin(t/(T+h)),0;
                  0,  0.1*cos(t/(T+h))];
    Omega(:,:,t) = [1, 0.3*sin(t/(T+h));
                   0.3*sin(t/(T+h)), 1];
        
    eta(pp+t,:) = epsilon(t,:) * chol(Omega(:,:,t));
        
    H_true(pp+t,:) = c(:,t)' + (V(pp+t-1,:).*V(pp+t-1,:))*(C(:,:,t))'+ H_true(pp+t-1,:)*(D(:,:,t))';
    V(t+pp,:) = eta(pp+t,:) * mpower(diag(H_true(pp+t,:)),0.5);
end
V = V(pp+1:end,:);
%再生成X根据VAR(p) X -p:1:(T+h)
X = [Xdiscard(end,:);zeros((T+h),m)];
A = zeros(m,m,(T+h));      % the time-varying autoregressive coefficients
A_d = zeros(m,m,(T+h));
gamma = zeros(13,(T+h));
gamma_d = zeros(13,(T+h));  % gamma 是所有系数向量，包括了Omega的部分
for t = 1:(T+h)
    A(:,:,t) = [0.5*exp(0.5*t/(T+h)-0.5)-0.1, -0.2*exp(t/(T+h)-1);
                -0.2*cos(pi*t/(T+h)), 0.6*exp(-t/(T+h)-0.5)];
    A_d(:,:,t) = [0.25*exp(0.5*t/(T+h)-0.5), -0.2*exp(t/(T+h)-1);
                  0.2*pi*sin(pi*t/(T+h)), -0.6*exp(-t/(T+h)-0.5)];
    gamma(:,t) =   [vec(A(:,:,t));c(:,t);vec(C(:,:,t));diag(D(:,:,t));0.3*sin(t/(T+h))];
    gamma_d(:,t) = [vec(A_d(:,:,t));c_d(:,t);vec(C_d(:,:,t));diag(D_d(:,:,t));0.3*cos(t/(T+h))];
        
    X(t+p,:) = X(t+p-1,:)*(A(:,:,t))' + V(t,:);
end
X = X(p+1:end,:);  % 取1:(T+h)
end 


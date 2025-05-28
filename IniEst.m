function [gamma_hat] = IniEst(X,p,tau,h_opt)
% Preliminary Estimation of GARCH
    [T,m] = size(X);
    Y = X.*X;
    p0 = ceil(2*T^(1/3));
    Y_lag = zeros(T-p0,m*p0);
    
    for t = 1:(T-p0)
        temp = [];
        for r = 1:p0
            temp = [temp,Y(p0+t-r,:)];
        end
        Y_lag(t,:) = temp;
    end
    
    Y = Y(p0+1:end,:);
    v_hat_ini = zeros(T-p0,m);              % the estimates of v_t
    
    for t = 1:(T-p0)
        tau_t = t/(T-p0);
        K = K_weight(T-p0,tau_t,h_opt);   % kernel weighting matrix
        v_hat_ini(t,:) = Y(t,:)-Y_lag(t,:)*inv(Y_lag'*K*Y_lag)*(Y_lag'*K*Y); % estimates of reduced form residuals
    end
        
    R = [eye(7),zeros(7,1);zeros(2,8);zeros(1,7),1];
    
    K = kron(K_weight(T-p0,tau,h_opt),eye(m));    
    
    Y_lag = Y_lag(:,1:m*p);
    v_hat_ini2 = [0,0;v_hat_ini(1:end-1,:)];
    X_new = [ones(T-p0,1),Y_lag,v_hat_ini2];
    X_gamma = kron(X_new,eye(m))*R;
    gamma_hat = inv(X_gamma'*K*X_gamma)*(X_gamma'*K*vec(Y'));
    
    c0_hat = gamma_hat(1:2,:);
    C1_hat = invVec(gamma_hat(3:6,:),2,2) + [gamma_hat(7,:),0;0,gamma_hat(8,:)];
    D1_hat = [-gamma_hat(7,:),0;0,-gamma_hat(8,:)];
    
%     c0d_hat = gamma_hat(9:10,:);
%     C1d_hat = invVec(gamma_hat(11:14,:),2,2) + [gamma_hat(15,:),0;0,gamma_hat(16,:)];
%     D1d_hat = [-gamma_hat(15,:),0;0,-gamma_hat(16,:)];
    
    gamma_hat = [c0_hat;vec(C1_hat);diag(D1_hat)];
    gamma_hat(gamma_hat < 0) = 0.05; 
%     gamma_hat = [gamma_hat;c0d_hat;vec(C1d_hat);diag(D1d_hat)];
    
%     h_hat_ini = zeros(T-p0,m);
%     eta_hat_ini = zeros(T-p0,m);
%     h_hat_ini(1,:) = c0_hat';
%     eta_hat_ini(1,:) = X(p0+1,:) * mpower( diag(h_hat_ini(1,:)),-0.5);
%     for t = 2:T-p0
%         h_hat_ini(t,:) = c0_hat' + (X(p0+t-1,:).*X(p0+t-1,:))*(C1_hat)'+ h_hat_ini(p+t-1,:)*D1_hat';
%         eta_hat_ini(t,:) = X(p0+t,:) * mpower( diag(h_hat_ini(t,:)),-0.5);
%     end
%     
%     % OLS estimates
%     Omega_ini = eta_hat_ini'*eta_hat_ini/(T-p0);
end


function [p] = NIC(X,P)
% order seltcion
[T,d] = size(X); % d denotes the dimension of x_t
IC = zeros(P,1);
for p=1:P
    X_lag = zeros((T-p),d*p+1);
    for t=1:(T-p)
        temp=1;
        for r=1:p
            temp=[temp,X(p+t-r,:)];
        end
        X_lag(t,:)=temp;
    end
    Y=X(p+1:T,:); % regresand

    h_opt = 2.34*sqrt(1/12)*(T-p)^(-0.2);      % optimal bandwidth

    % local linear estimation
    eta_hat=zeros(T-p,d);              % the estimates of eta_t
    for t=1:(T-p)
        tau_t=t/(T-p);
        Z=Z_data(X_lag,tau_t,h_opt);
        K=K_weight(T-p,tau_t,h_opt);   % kernel weighting matrix
        A_hat=([eye(d*p+1) zeros(d*p+1)]*inv(Z'*K*Z)*(Z'*K*Y))';
        eta_hat(t,:)=Y(t,:)-X_lag(t,:)*A_hat';
    end
    chi_T=[h_opt^4,h_opt^2*(log(T)/(T*h_opt))^(0.5),(log(T)/(T*h_opt))];
    IC(p)=log(trace(eta_hat'*eta_hat)/(T-p))+p*log(log(T*h_opt))*max(chi_T);
end
p=find(IC==min(IC)); 
end


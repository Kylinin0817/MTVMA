function [Loglf] = Likelihood_VAR_GARCH2(X,para,p,tau,h)
% the likelihood function of VAR-GARCH process ，p 为diag()内的具体的滞后阶数；
%输入X是T,m;用的lf是T-p，m，主要是考虑了方差部分的p阶滞后
[T,m] = size(X);
A1 = invVec(para(1:4,:),2,2);
c0 = exp(para(5:6,:));
C1 = invVec(normcdf(para(7:10,:)),2,2);
D1 = [normcdf(para(11,:)),0;0,normcdf(para(12,:))];
Omega = [1,2*normcdf(para(13,1))-1;2*normcdf(para(13,1))-1,1];

A1d = invVec(para(14:17,:),2,2);
c0d = para(18:19,:);
C1d = invVec(para(20:23,:),2,2);
D1d = [para(24,:),0;0,para(25,:)];
Omegad = [0,para(26,1);para(26,1),0];

lf = zeros(T,1);  % store the probability density of x_t

H = zeros(T,m);
V = zeros(T,m);

if abs(1/T-tau) <= h
    H(1,:) = (c0+c0d*(1/T-tau)/h)';
    V(1,:) = X(1,:);
    %lf(1,1) = V(1,:)*inv(mpower(diag(H(1,:)),0.5)*(Omega+Omegad*(1/T-tau)/h)*mpower(diag(H(1,:)),0.5))*V(1,:)' + log(det(mpower(diag(H(1,:)),0.5)*(Omega+Omegad*(1/T-tau)/h)*mpower(diag(H(1,:)),0.5)));
end

for t = 2:T    
    V(t,:) = X(t,:)-X(t-1,:)*A1';
    if t >= p+1
        tau_t = (t-p)/(T-p);
        if abs(tau_t-tau) <= h
            V(t,:) = X(t,:) - X(t-1,:)*(A1+A1d*(tau_t-tau)/h)';
            if p==1
                aa = (V(t-1,:).*V(t-1,:))';
            elseif p>1
                %aa = mpower(D1+D1d*(tau_t-1)/h,p-2)*(C1+C1d*(tau_t-1)/h)*(V(t-p+1,:).*V(t-p+1,:))'+mpower(D1+D1d*(tau_t-1)/h,p-1)*(C1+C1d*(tau_t-1)/h)*(V(t-p,:).*V(t-p,:))';
                aa = (V(t-1,:).*V(t-1,:))' + mpower(D1+D1d*(tau_t-1)/h,p-1)*(C1+C1d*(tau_t-1)/h)*(V(t-p,:).*V(t-p,:))';
            end
            %for  i = 1:p-1
            %   aa = aa + mpower(D1+D1d*(tau_t-1)/h,i)*(C1+C1d*(tau_t-1)/h)*(V(t-1-i,:).*V(t-1-i,:))'; 
            %end
            H(t,:) = (c0+c0d*(tau_t-tau)/h)'* pinv(eye(m)-D1)+ aa';
            % 上面  这里Vt-1 滞后阶数为相应候选模型的滞后阶数
            lf(t,1) = V(t,:)*pinv(mpower(diag(H(t,:)),0.5)*(Omega+Omegad*(tau_t-tau)/h)*mpower(diag(H(t,:)),0.5))*V(t,:)' + log(det(mpower(diag(H(t,:)),0.5)*(Omega+Omegad*(tau_t-tau)/h)*mpower(diag(H(t,:)),0.5)));
        end
    end
end

K = K_weight(T,tau,h);   % kernel weighting matrix
Loglf = sum(K*lf)/(T-p);
end


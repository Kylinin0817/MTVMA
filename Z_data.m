function [Z] = Z_data(X,tau,h )
% output Z^\top=(z_1,...,z_T) z_t=(x_t^\top,(\tau_t-\tau)/h*x_t^\top)^\top
[T,k]=size(X);
tau_t=[1:T]/T;
Z=zeros(T,2*k);
for t=1:T 
    Z(t,:)=[X(t,:) X(t,:)*(tau_t(t)-tau)/h];
end

end


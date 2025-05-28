function [K] = K_h(tau_t,tau,h)
% Epanechnikov kernel 核函数，tau_t和tau距离的权重
if abs((tau-tau_t)/h)<=1
    K=0.75*(1-power((tau-tau_t)/h,2));
else
    K=0;
end
end


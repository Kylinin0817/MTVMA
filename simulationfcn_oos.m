function [RMSE] = simulationfcn_oos(T,h,Pmax)
rng(0);
m = 2;
d = m*(m+1)/2;
K = 100; %重复试验次数
v = 0.6*T;

pre_error = zeros(K, m, 9); % 每一次重复次数都对应一个m维的误差，i.e.每次实验每个分量都有误差
RMSE = zeros(m, 9); %样本外的RMSE，测试集只有一个样本

parfor k = 1:K %并行
    disp(k)
    p0 = 1; %初始AR滞后阶数
    pp0 = 1; %初始ARCH滞后阶
    % 1. 依据真实生成过程生成样本T个
    [X, gamma, gamma_d] = DGP2(T,p0,pp0,h); %生成T个符合真实数据生成过程的样本
    phat = NIC(X(1:T,:), Pmax); % NIC准则估计的滞后阶数
    Y_pred = zeros(m, Pmax);  % 在Pmax个候选模型下，测试集上的1个样本的估计
    Sigma_hat = zeros(m,m,T+h-Pmax-v,Pmax); % Pmax个候选模型，有Pmax个协方差矩阵的估计，记录T-Pmax每个时刻的协方差矩阵m,m的估计
    eta_hat = zeros(T+h-Pmax-v, m, Pmax); % Pmax个候选模型，每个候选模型的eta_hat是T-Pmax,m大小的，记录残差
    AIC = zeros(1, Pmax); 
    BIC = zeros(1, Pmax);
    HQ = zeros(1, Pmax);
    ww_sAIC = zeros(1, Pmax);
    ww_sBIC = zeros(1, Pmax);
    ww_sHQ = zeros(1, Pmax);
    %% 2. 不删点情况下候选模型的估计，时刻是最后一个时刻
    for p = 1:Pmax  % 这里等运行完改回来 1：Pmax
        % 生成滞后样本
        % p+1 ~ p,...,1
        % T  ~ T-1,...,T-p
        % X 是T+h个样本，用于训练的X_p是p+1:T,用于测试的是T+1:T+h,滞后样本是p+1:T+h
        X_lag = X(p:T+h-1,:);
        % 1.1. 对于不同的滞后阶数，生成对应的滞后矩阵
        X_p = X((p+1):T+h,:); % 被预测样本
        h_opt = 2.34*sqrt(1/12)*(T+h-p)^(-0.2);
        tt = T+h-p;
        tau = 1;
        % gamma，Omega 初始化
        %gamma_p = zeros((T-p),13);  % 真实数据生成过程对应的系数，有p0+m*m,样本量为T-p个，用上了所有的样本，但是当用前向的方法，要设一个初始时间v
        %gamma_d_p = zeros((T-p),13);
        
        % Omega_p = zeros((T-p),d);
        % 2.2. 估计未知参数；注意不是所有点的未知参数都有估计，只有部分点，因为要保证可优化，t不能太小；这里有问题；因为不能得到T-p个时刻的参数的估计；只能得到n个样本点
        gamma_true = gamma(:,tt);
        gamma_d_true = gamma_d(:,tt);
        % 初始值是真实值做了个变换
        para0 = [gamma_true(1:4);log(gamma_true(5:6));norminv(gamma_true(7:12));norminv((gamma_true(13)+1)/2);gamma_d_true(:)*h_opt];
        [para1,~,~,~,~,~] = fminunc(@(para)Likelihood_VAR_GARCH2(X_p,para,p,tau,h_opt),para0,optimset('Display','off','MaxIter', 1000));
        A1_t_p = invVec(para1(1:4,:),2,2);
        Y_fit_p = X_lag(tt,:)*A1_t_p'; % mu的话，只用到了A1
        
        Y_pred(:,p) = Y_fit_p';  % 最后一个时刻的估计值，在时变多元因果过程的模型下
    end
   
    %% 3. 删点情况下的参数的估计，拟合值v+1:T+h-p
    for p = 1:Pmax  % 这里等运行完改回来 1：Pmax
        % 生成滞后样本
        % p+1 ~ p,...,1
        % T  ~ T-1,...,T-p
        X_lag = X(p:T+h-1,:);
        X_p = X((p+1):T+h,:); % 被预测样本
        % X_lag_h = X_lag(1:T-p,:); % regressor
        h_opt = 2.34*sqrt(1/12)*(T+h-p)^(-0.2);
        Y_fit_p = zeros((T+h-p-v), m);
        eta_hat_p = zeros((T+h-p-v), m);
        % gamma，Omega 初始化
        %gamma_p = zeros((T-p),13);  % 真实数据生成过程对应的系数，有p0+m*m,样本量为T-p个，用上了所有的样本，但是当用前向的方法，要设一个初始时间v
        %gamma_d_p = zeros((T-p),13);
        gamma_true = zeros(13,T+h-p-v);    % true value vec(A1,c0,C1,D1,rho)
        gamma_d_true = zeros(13,T+h-p-v);  % true value vec(A1,c0,C1,D1,rho)
        Sigma_p = zeros(m,m,T+h-p-v);  % 用于记录T-p个样本下的协方差矩阵的vech所以是每行都是1*d
        H_p = zeros(T+h-p-v,2);
        V_p = zeros(T+h-p-v,2);
        % Omega_p = zeros((T-p),d);
        % 2.2. 估计未知参数；注意不是所有点的未知参数都有估计，只有部分点，因为要保证可优化，t不能太小；这里有问题；因为不能得到T-p个时刻的参数的估计；只能得到n个样本点
        
        % 下面通过QMLE估计gamma，Omega
        % 首先给一个gamma和Omega的初始估计
        % QMLE
        for t = v+1:T+h-p
            gamma_true(:,t) = gamma(:,t);
            gamma_d_true(:,t) = gamma_d(:,t);
            tau = (t-v)/(T+h-p-v);
            %所以初始值就是真实值
            para0 = [gamma_true(1:4,t);log(gamma_true(5:6,t));norminv(gamma_true(7:12,t));norminv((gamma_true(13,t)+1)/2);gamma_d_true(:,t)*h_opt];
            [para1,~,~,~,~,~] = fminunc(@(para)Likelihood_VAR_GARCH2_forward(X_p,para,p,h_opt,t),para0,optimset('Display','off','MaxIter', 1000));
            %para1:[gamma;gammad;omega;omegad]
            %dim   [p0+m*m;p0+m*m;d;d]
            % Think:  在换一换tau_set，把样本点选多一点，试一试！
            % 不能得到T-p个时刻的权重的值，只能得到n个时刻的权重的值，那这n个时刻，对应的样本的估计为：
            % 2.3 估计Y_fit_p :第p个候选模型下的，tau时刻的mu的估计
            A1_t_p = invVec(para1(1:4,:),2,2);
            c0_t_p = exp(para1(5:6,:));
            C1_t_p = invVec(normcdf(para1(7:10,:)),2,2);
            D1_t_p = [normcdf(para1(11,:)),0;0,normcdf(para1(12,:))];
            Omega_t_p = [1,2*normcdf(para1(13,1))-1;2*normcdf(para1(13,1))-1,1];
            %每一时刻的协方差估计
            if t-v==1
                H_p(t-v,:) = c0_t_p';
                V_p(t-v,:) = X_p(t,:);
            elseif t-v>1
                V_p(t-v,:) = X_p(t,:) - X_p(t-1,:)*A1_t_p';
                H_p(t-v,:) = c0_t_p' + (V_p(t-v-1,:).*V_p(t-v-1,:))*C1_t_p' + H_p(t-v-1,:)*D1_t_p';
            end
            Sigma_p(:,:,t-v) = mpower(diag(H_p(t-v,:)),0.5) * Omega_t_p * mpower(diag(H_p(t-v,:)),0.5);
            Y_fit_p(t-v,:) = X_lag(t,:)*A1_t_p'; % mu的话，只用到了A1
            eta_hat_p(t-v,:) = X_p(t,:) - Y_fit_p(t-v,:);  
            % 也就是只有这T-p个时间点都有残差值
        end
        eta_hat(:,:,p) = eta_hat_p((1+Pmax-p):end,:);  % 将每个p候选模型的样本个数都统一成T-Pmax
        Sigma_hat(:,:,:,p) = Sigma_p(:,:,(1+Pmax-p):end); %归一成T-Pmax个样本
        
    end
    %% 3. 模型的权重选取准则
    %PVEC = 1:Pmax;
    h_opt_ma = 2.34*sqrt(1/12)*(T+h-Pmax)^(-0.2);
    QQ = zeros(Pmax, Pmax);
    K_ma = K_weight((T-Pmax-v), 1, h_opt_ma); % 这里为什么tau取1呢
    %这里权重选取准则得改！！!!!!!!!
    for j = 1:(T-Pmax-v)  % eta_hat: T-Pmax,m,Pmax; Omega_hat: m,m,Pmax; K_ma: T-Pmax, T-Pmax
        QQ = QQ + K_ma(j, j) * squeeze(eta_hat(j,:,:))' * pinv(squeeze(Sigma_hat(:,:,j,Pmax))) * squeeze(eta_hat(j,:,:));
    end    % 感觉这里应该是K_ma(j,T-Pmax)，这里用T-Pmax是因为是针对tau=1时刻的样本点的权重的估计。
    % 不同时刻对应的条件均值和协方差的估计都是不同的;
    %%% ------------Q:和TVMA不同的是，这里的协方差是时变的，而TVMA中的是用的最新的协方差？？？？哪个更好-------------

    QQ = (QQ + QQ')/(T-Pmax-v);  % 这里加转置没有问题把？
    %B = 2 * log((T-Pmax) * h_opt_ma) * m^2 * PVEC';
    % 没有惩罚项的版本
    % 1/2w^T Omega w + f^T w
    ww = quadprog(QQ, zeros(Pmax,1), zeros(1, Pmax), 0, ones(1, Pmax), 1, zeros(Pmax, 1), ones(Pmax, 1));
    ww = ww(ww > 0);
    ww = ww / sum(ww);

    for p = 1:Pmax
        Omega_residual = eta_hat(:,:,p)' * eta_hat(:,:,p) / (T+h-Pmax-v);
        AIC(1, p) = log(det(squeeze(Omega_residual))) + 2 * p * m^2 / (T+h-Pmax-v);
        BIC(1, p) = log(det(squeeze(Omega_residual))) + log(T+h-Pmax-v) * p * m^2 / (T+h-Pmax-v);
        HQ(1, p) = log(det(squeeze(Omega_residual))) + 2 * log(log(T+h-Pmax-v)) * p * m^2 / (T+h-Pmax-v);
    end

    [~, loc_AIC] = min(AIC);
    [~, loc_BIC] = min(BIC);
    [~, loc_HQ] = min(HQ);

    ww_sAIC(1,:) = exp(-AIC(1,:) / 2);
    ww_sBIC(1,:) = exp(-BIC(1,:) / 2);
    ww_sHQ(1,:) = exp(-HQ(1,:) / 2);
    ww_sAIC(1,:) = ww_sAIC(1,:) / sum(ww_sAIC(1,:));
    ww_sBIC(1,:) = ww_sBIC(1,:) / sum(ww_sBIC(1,:));
    ww_sHQ(1,:) = ww_sHQ(1,:) / sum(ww_sHQ(1,:));

    Y_true = X(end,:);
    Y_pred_ma = Y_pred * ww;
    Y_pred_NIC = Y_pred(:, phat);
    Y_pred_AIC = Y_pred(:, loc_AIC);
    Y_pred_BIC = Y_pred(:, loc_BIC);
    Y_pred_HQ = Y_pred(:, loc_HQ);
    Y_pred_sAIC = Y_pred * ww_sAIC';
    Y_pred_sBIC = Y_pred * ww_sBIC';
    Y_pred_sHQ = Y_pred * ww_sHQ';
    Y_pred_SA = Y_pred * ones(Pmax,1)/Pmax;

    pre_error_ma(k,:) = Y_true - Y_pred_ma';
    pre_error_NIC(k,:) = Y_true - Y_pred_NIC';
    pre_error_AIC(k,:) = Y_true - Y_pred_AIC';
    pre_error_BIC(k,:) = Y_true - Y_pred_BIC';
    pre_error_HQ(k,:) = Y_true - Y_pred_HQ';
    pre_error_sAIC(k,:) = Y_true - Y_pred_sAIC';
    pre_error_sBIC(k,:) = Y_true - Y_pred_sBIC';
    pre_error_sHQ(k,:) = Y_true - Y_pred_sHQ';
    pre_error_SA(k,:) = Y_true - Y_pred_SA';
end




pre_error(:,:,1) = pre_error_ma;
pre_error(:,:,2) = pre_error_NIC;
pre_error(:,:,3) = pre_error_AIC;
pre_error(:,:,4) = pre_error_BIC;
pre_error(:,:,5) = pre_error_HQ;
pre_error(:,:,6) = pre_error_sAIC;
pre_error(:,:,7) = pre_error_sBIC;
pre_error(:,:,8) = pre_error_sHQ;
pre_error(:,:,9) = pre_error_SA;

for i = 1:9
    for j = 1:m
        RMSE(j, i) = sqrt(pre_error(:, j, i)' * pre_error(:, j, i) / K);
    end
end
end
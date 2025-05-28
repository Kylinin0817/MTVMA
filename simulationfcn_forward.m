function [RMSE] = simulationfcn_forward(T,h,Pmax)
rng(0);
m = 2;
d = m*(m+1)/2;
K = 1000; %重复试验次数
v = 3*T^(1/2);
pre_error = zeros(K, m, 9); % 每一次重复次数都对应一个m维的误差，i.e.每次实验每个分量都有误差
RMSE = zeros(m, 9); %样本外的RMSE，测试集只有一个样本
tau_set = 0.2:0.05:0.8;
n = size(tau_set,2);
parfor k = 1:K %并行
    disp(k)
    p0 = 1; %初始AR滞后阶数
    pp0 = 1; %初始ARCH滞后阶
    % 1. 依据真实生成过程生成样本T个
    [X, gamma, gamma_d] = DGP2(T,p0,pp0); %生成T个符合真实数据生成过程的样本
    phat = NIC(X(1:T,:), Pmax); % NIC准则估计的滞后阶数
    Y_pred = zeros(m, Pmax);  % 在Pmax个候选模型下，测试集上的1个样本的估计
    Sigma_hat = zeros(m,m,T-Pmax, Pmax); % Pmax个候选模型，有Pmax个协方差矩阵的估计，记录T-Pmax每个时刻的协方差矩阵m,m的估计
    eta_hat = zeros(T-Pmax, m, Pmax); % Pmax个候选模型，每个候选模型的eta_hat是T-Pmax,m大小的，记录残差
    AIC = zeros(1, Pmax); 
    BIC = zeros(1, Pmax);
    HQ = zeros(1, Pmax);
    ww_sAIC = zeros(1, Pmax);
    ww_sBIC = zeros(1, Pmax);
    ww_sHQ = zeros(1, Pmax);
    %% 2. 候选模型的估计
    for p = 1:Pmax  % 这里等运行完改回来 1：Pmax
        % 生成滞后样本
        % p+1 ~ p,...,1
        % T  ~ T-1,...,T-p
        X_lag = X(p:T-1,:);
        X_p = X((p+1):T,:); % 被预测样本
        % X_lag_h = X_lag(1:T-p,:); % regressor
        h_opt = 2.34*sqrt(1/12)*(T-p)^(-0.2);
        Y_fit_p = zeros((T-p-v), m);
        eta_hat_p = zeros((T-p-v), m);
        % gamma，Omega 初始化
        %gamma_p = zeros((T-p),13);  % 真实数据生成过程对应的系数，有p0+m*m,样本量为T-p个，用上了所有的样本，但是当用前向的方法，要设一个初始时间v
        %gamma_d_p = zeros((T-p),13);
        gamma_true = zeros(13,T-p);    % true value vec(A1,c0,C1,D1,rho)
        gamma_d_true = zeros(13,T-p);  % true value vec(A1,c0,C1,D1,rho)
        Sigma_p = zeros(m,m,T-p-v);  % 用于记录T-p个样本下的协方差矩阵的vech所以是每行都是1*d
        H_p = zeros(T-p-v,2);
        V_p = zeros(T-p-v,2);
        % Omega_p = zeros((T-p),d);
        % 2.2. 估计未知参数；注意不是所有点的未知参数都有估计，只有部分点，因为要保证可优化，t不能太小；这里有问题；因为不能得到T-p个时刻的参数的估计；只能得到n个样本点
        
        % 下面通过QMLE估计gamma，Omega
        % 首先给一个gamma和Omega的初始估计
        % QMLE
        for t = v+1:T-p
            disp(t);
            gamma_true(:,t) = gamma(:,t);
            gamma_d_true(:,t) = gamma_d(:,t);
            tau = (t-v)/(T-p-v);
            %所以初始值就是真实值
            para0 = [gamma_true(1:4,t);log(gamma_true(5:6,t));norminv(gamma_true(7:12,t));norminv((gamma_true(13,t)+1)/2);gamma_d_true(:,t)*h_opt];
            [para1,~,~,~,~,~] = fminunc(@(para)Likelihood_VAR_GARCH2(X,para,p,tau,h_opt,t),para0,optimset('Display','off','MaxIter', 1000));
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
        
        % MA权重选择
        %for t = 1:n   %这里设置n个时间点的样本来估计参数是为了不让t太小，否则无法求优
        %    tau = tau_set(1,t);
            % gamma_true(t,:) = gamma_p(ceil(tau*T),:)这里不需要gamma_true因为我们不讨论覆盖率
       %     [gamma_ini,Omega_ini] = IniEst(X,p0,tau);  % 输入是训练样本和当前滞后阶和当前时间t/T-p
            %输出是[2(p0+m^2)*1;2d*1]=[12;6]
            %也就是说初步估计的参数是VARMA(2,1)的参数，不是转化成VAR(\infty)对应的参数，VAR对应的参数可以用VARMA的参数求出来
       %     para0 = [gamma_ini;Omega_ini]; %进行优化的初始参数
       %     [para1,~,~,~,~,~] = fminunc(@(para)Likelihood_VARMA(X,para,p,tau,h),para0,optimset('Display','off','MaxIter', 10000));
       %     gamma_p(ceil(tau*T),:) = para1(1:(p0+m*m),1)'; % 1,p+m*m;为gamma的估计
       %     Omega_p(ceil(tau*T),:) = para1(p0+m*m+1:p0+m*m+d,1)';
            % Think: !!!!! 在换一换tau_set，把样本点选多一点，试一试！
            % 不能得到T-p个时刻的权重的值，只能得到n个时刻的权重的值，那这n个时刻，对应的样本的估计为：
            % 2.3 估计Y_fit_p :第p个候选模型下的，tau时刻的mu的估计
       %     a1 = para1(1,1);
       %     a2 = para1(2,1);
       %     B1 = invVec(para1(3:6,1),m,m);
       %     a1d = para1(7,1);       
       %     a2d = para1(8,1);
       %     B1d = invVec(para1(9:12,1),m,m);
       %     Omega = [para1(13,1),para1(14,1);para1(14,1),para1(15,1)];
       %     Omegad = [para1(16,1),para1(17,1);para1(17,1),para1(18,1)];
       %     Gamma = zeros(m,m*p);
       %     Gamma(:,1:m) = (a1)*mpower(-(B1),0) - mpower(-(B1),1);
       %     if p >=2 
       %         for j = 2:p
       %             Gamma(:,(j-1)*m+1:j*m) = (a1)*mpower(-(B1),j-1) + (a2)*mpower(-(B1),j-2) - mpower(-(B1),j);
       %         end
       %     end
            %mu = Gamma(:,1:(m*p))*vec(X(t-1:-1:(t-p),:)');
       %     Y_fit_p(ceil(tau*T),:) = Gamma(:,1:(m*p))*vec(X(ceil(tau*T)-1:-1:(ceil(tau*T)-p),:)');
       %     eta_hat_p(ceil(tau*T),:) = X_p(ceil(tau*T),:) - Y_fit_p(ceil(tau*T),:);
            % 也就是只有这n个时间点有残差值
       % end
        
        % 3. 权重选取准则 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        %y_pred_tem = zeros(1, h*m);
        %for i = 1:h
        %    if i <= p
        %        y_lagged = [X_lag(end,1), y_pred_tem(:,(end-m*(i-1)+1):end), X_lag(end,2:(1+m*p-m*i+2))];
        %    else
        %        y_lagged = [1, y_pred_tem(:, (end-m*(i-1)+1):(end-m*(i-p-1)))];
        %    end
        %    y_pred_tem(:,(end-m*i+1):(end-m*i+m)) = y_lagged; %* A_hat_p(:,:,T-p)';
        %end
        
        %Y_pred(:,p) = y_pred_tem(1,1:m)'; %m,1
        Y_pred(:,p) = Y_fit_p(T-p-v,:)';  % 最后一个时刻的估计值，在时变多元因果过程的模型下
        %%tau_t_omega = 1;  % 因为是将最后一个时刻作为了测试样本，所以tau=1
        %%K_Ome = K_weight(T-p-v, tau_t_omega, h_opt);   % 其他时刻的样本对估计测试样本起到的作用矩阵；算第p个候选模型下的协方差的估计时用的样本点是最新的样本
        %Omega_hat(:,:,p) = eta_hat_p(:,:)' * K_Ome * eta_hat_p(:,:) / (sum(diag(K_Ome)));  %eta_hat_p是T-p，m维的，Omega_hat是m,m维的，每一个候选模型都有，有p个；但在权重选择时，只用Pmax对应的协方差矩阵。
        % 这个Omega_hat是用最新的几个样本来估计测试样本时刻的协方差矩阵(m,m)×p；但是我们有协方差矩阵的估计，在QMLE中，所以这里要改一下
        %Omega_hat(:,:,:,p) = invVec(Omega_p(T-p,:),m,m);  % 算的是 p滞后阶数下的  T-p时刻，也就是最后一个测试样本对应的时刻的  (m,m)维的协方差矩阵的
        eta_hat(:,:,p) = eta_hat_p((1+Pmax-p):end,:);  % 将每个p候选模型的样本个数都统一成T-Pmax
        %%协方差的估计
        
        Sigma_hat(:,:,:,p) = Sigma_p(:,:,(1+Pmax-p):end); %归一成T-Pmax个样本
        % 这样，就每个模型p，每个时刻T-Pmax，都有一个m×m的协方差的估计，记录在Omega_hat中
        %         每个模型p，每个时刻T-Pmax，都有一个m 的残差的估计，记录在eta_hat中
    end
    %% 3. 模型的权重选取准则
    %PVEC = 1:Pmax;
    h_opt_ma = 2.34*sqrt(1/12)*(T-Pmax)^(-0.2);
    QQ = zeros(Pmax, Pmax);
    K_ma = K_weight((T-Pmax-v), 1, h_opt_ma); % 这里为什么tau取1呢
    %这里权重选取准则得改！！!!!!!!!
    for j = 1:(T-Pmax-v)  % eta_hat: T-Pmax,m,Pmax; Omega_hat: m,m,Pmax; K_ma: T-Pmax, T-Pmax
        QQ = QQ + K_ma(j, j) * squeeze(eta_hat(j,:,:))' * inv(squeeze(Sigma_hat(:,:,j,Pmax))) * squeeze(eta_hat(j,:,:));
    end    % 感觉这里应该是K_ma(j,T-Pmax)，这里用T-Pmax是因为是针对tau=1时刻的样本点的权重的估计。
    % 不同时刻对应的条件均值和协方差的估计都是不同的;
    %%% ------------Q:和TVMA不同的是，这里的协方差是时变的，而TVMA中的是用的最新的协方差？？？？哪个更好-------------

    QQ = QQ + QQ';  % 这里加转置没有问题把？
    %B = 2 * log((T-Pmax) * h_opt_ma) * m^2 * PVEC';
    % 没有惩罚项的版本
    % 1/2w^T Omega w + f^T w
    ww = quadprog(QQ, zeros(Pmax,1), zeros(1, Pmax), 0, ones(1, Pmax), 1, zeros(Pmax, 1), ones(Pmax, 1));
    ww = ww(ww > 0);
    ww = ww / sum(ww);

    for p = 1:Pmax
        Omega_residual = eta_hat(:,:,p)' * eta_hat(:,:,p) / (T-Pmax-v);
        AIC(1, p) = log(det(squeeze(Omega_residual))) + 2 * p * m^2 / (T-Pmax-v);
        BIC(1, p) = log(det(squeeze(Omega_residual))) + log(T-Pmax-v) * p * m^2 / (T-Pmax-v);
        HQ(1, p) = log(det(squeeze(Omega_residual))) + 2 * log(log(T-Pmax-v)) * p * m^2 / (T-Pmax-v);
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
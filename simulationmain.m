clc;
clear;
%result1 = zeros(8,9);
%result2 = zeros(8,9);
result3 = zeros(8,9);
poolobj = gcp('nocreate');
if isempty(poolobj)
    poolsize = 0;
    CoreNum = 8;                    % 设置CPU核心数量
    parpool('local', CoreNum);
else
    poolsize = poolobj.NumWorkers;
    disp('Already initialized');    % 并行环境已启动
end
h=1;   % 只有一个点
%[result1(1:2,:)] = simulationfcn_oos(200,h,5);
%[result2(1:2,:)] = simulationfcn_oos(200,h,10);
[result3(1:2,:)] = simulationfcn_oos(200,h,10);
delete(gcp('nocreate'));
%result1 = result1 ./ result1(:,3);
%result2 = result2 ./ result2(:,3);
result3 = result3 ./ result3(:,3);

%result1(:,3) = [];
%result2(:,3) = [];
result3(:,3) = [];

columnNames = {'ma', 'nic', 'bic', 'hq', 'saic', 'sbic', 'shq', 'sa'};


%resultTable = array2table(result1, 'VariableNames', columnNames);
%resultTable2 = array2table(result2, 'VariableNames',columnNames);
resultTable3 = array2table(result3, 'VariableNames',columnNames);


%writetable(resultTable, 'onlyone-pth-lag_VAR_GARCH.xlsx', 'Sheet', 'results1');
%writetable(resultTable2, 'onlyone-pth-lag_VAR_GARCH_result2.xlsx', 'Sheet', 'results2');
writetable(resultTable3, 'onlyone-pth-lag_VAR_GARCH_result3.xlsx', 'Sheet', 'results3');

function [ label_pred, run_time, perf ] = Lite_mfCGL(data_train, label_train, data_test, label_test)
%   label_train:    nSamp * nLab
%   data_train:     nSamp * nFeat

featr = [data_train*1 ones(size(data_train,1), 1)];
feats = [data_test*1 ones(size(data_test,1), 1)];

% Default parameters
rho1 = 0.01;
rho2_tmp = logspace(-3,-1,30);
switch ceil( size(label_train,2) / 10 )
    case 1 % Scene, LabelMe
        rho2 = rho2_tmp(18);
    case 2 % PASCAL07, PASCAL12
        rho2 = rho2_tmp(8);
    otherwise
        rho2 = 0.01;
end

% training
star_t = tic;
[B,A] = CGL_Learning(featr,label_train,rho1,rho2);
run_time.train = toc(star_t);

% prediction
star_t = tic;
label_pred = double(CGL_Inference(feats,B,A)>0.5);
run_time.test = toc(star_t);

% Calculate performance if required
if nargout == 3
    perf = get_perform(label_test,label_pred);
end
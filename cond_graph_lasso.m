function [B,A] = cond_graph_lasso(X,Y,rho1,rho2)
%%% Conditional Graphical Lasso (CGL) for multilabel data
%      p(y|x) \proportion_to exp{x*B*y + 0.5*y'*A(x)*y};
%  i.e., p(y|x) \proportion_to exp{ \sum_i yi*<x,Bi> + \sum_{i<j} <A_ij,x>*yi*yj };
% input: X n by (p-1)+1, where p-1 is feature dimensionality, and the added
%        1 is for bias parameter
%        Y n by m, 0-1 matrix. where m is label number;
%        rho is weight parameter
% output: B p by m, the model
%         A p by m*m, each column can be reshaped into a symmetric matrix, with diag(A) = 0,
%         <A_ij,x> models the correlation between y_i and y_j

star_t = tic;
m = size(Y,2);
[n,p] = size(X);

loc = (0:m-1)*m + (1:m); % diagnal index of label adj matrix.

xB0 = zeros(p,m) ;        % initialize B
yB = xB0;
xA0 = zeros(p,m*m); xA0(:,loc) = 0; % initialize A
yA = xA0;

maxIter = 500;  %max iteration
t = 1/((sum(X(:).^2))/n)*30;          %step size

tol = 1e-3;
tc = 1;

Y_ = -((-1).^Y); % {0,1} --> {-1,1}
X = full(X);

for ii = 1:maxIter
    
    %%%%%% calculate objective and gradient
    % objetive on yB and yA
    eta = X*yB; % n by m
    temp = reshape(X*yA,[n,m,m]); % n by m by m
    xi = cell2mat(arrayfun(@(kk)squeeze(temp(kk,:,:))*Y(kk,:)', 1:n,'un',0))';  % n by m
    T = eta + xi; % n by m
    
    Y_T = Y_.*T; % n by m
    
    fv(ii) = 1/n*sum(sum( log(1+exp(-Y_T)) )) +...
        rho1*sum(sum( yB(1:end-1,:).^2 )) +...
        rho2*sum(sum( yA(1:end-1,:).^2 )) + rho2*sqrt(m)*sum(sqrt(sum( yA(1:end-1,:).^2 ))); % rho2*sum(sum(abs(yA(:,1:end-1))));
    
    % gradient on yB and yA
    Xi = -Y_./(1+exp(Y_T)); % n by m
    H_B = 1/n*X'*Xi;
    H_B(1:end-1,:) = H_B(1:end-1,:) + 2*rho1*yB(1:end-1,:);
    
    H_A = 1/n*X' * cell2mat(arrayfun(@(kk)reshape(Xi(kk,:)'*Y(kk,:),[m*m,1]), 1:n,'un',0))';
    H_A(1:end-1,:) = H_A(1:end-1,:) + 2*rho2*yA(1:end-1,:);
    H_A(:,loc) = 0;
    H_Atp = cell2mat(arrayfun(@(kk)reshape( reshape(H_A(kk,:),[m,m])',[m*m,1] ), 1:p,'un',0))';
    H_A = (H_A + H_Atp)/2;
    
    %  gradient descent updating
    xB = yB - t*H_B;
%     xBtmp1 = abs(xB);
%     xBtmp1 = max(0,xBtmp1 - t*rho1);
%     xBtmp2 = sign(xB);
%     xB = xBtmp1.*xBtmp2;
    
    xA = yA - t*H_A;
    for gi = 1:size(xA,2) % iterate each group
        if norm(xA(:,gi))~=0; xA(:,gi) = shrinkage(xA(:,gi), t*sqrt(m)*rho2); end
    end
%     xAtmp1 = abs(xA);
%     xAtmp1 = max(0,xAtmp1 - t*rho2);
%     xAtmp2 = sign(xA);
%     xA = xAtmp1.*xAtmp2;
    
    tc_ = (1+sqrt(1+4*tc^2))/2;
    if ii>20 && abs((fv(ii)-fv(ii-1))/fv(ii))<tol
        conv_t = toc(star_t);
        disp(['converged with ' num2str(conv_t) ' seconds']);
        B = xB;
        A = xA;
        return
    end
    yB = xB + (tc-1)/(tc_)*(xB-xB0);
    yA = xA + (tc-1)/(tc_)*(xA-xA0);
    tc = tc_;
    xB0 = xB;
    xA0 = xA;

end
B = xB;
A = xA;
if ii >= maxIter
    conv_t = toc(star_t);
    disp(['not converged within ' num2str(conv_t) ' seconds']);
end

% === sub-functions
function z = shrinkage(x, kappa)
z = max(0, 1 - kappa/norm(x))*x;

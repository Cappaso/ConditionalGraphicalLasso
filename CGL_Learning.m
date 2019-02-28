function [B,A] = CGL_Learning(X,Y,rho1,rho2)
% Function to implement CGL Learning
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
loc2 = reshape(reshape(1:m*m,m,m)',m*m,1); % index of label adj matrix transpose.

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
    
    qY = CGL_Inference( X,yB,yA,5 );
    
    % gradient on yB and yA
    EqY = (Y_==1).*qY + (-1)*(Y_==-1).*(1-qY); % n by m. Finally fixed this bug!
    H_B = X'*(-Y_ + EqY)/(2*n); % p by m
    H_B(1:end-1,:) = H_B(1:end-1,:) + 2*rho1*yB(1:end-1,:);
    
    Y_2 = cell2mat(arrayfun(@(kk)reshape(Y_(kk,:)'*Y_(kk,:),[m*m,1]), 1:n,'un',0))'; % n by m*m
    EqY2 = cell2mat(arrayfun(@(kk)reshape(EqY(kk,:)'*EqY(kk,:),[m*m,1]), 1:n,'un',0))'; % n by m*m
    H_A = X'*(-Y_2 + EqY2)/(2*n); % p by m*m
    H_A(1:end-1,:) = H_A(1:end-1,:) + 2*rho2*yA(1:end-1,:);
    H_A(:,loc) = 0;
    H_A = (H_A + H_A(:,loc2))/2; % symmetrize
    
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
    
    fv(ii) = max(norm(xB(:)-xB0(:))/norm(xB0(:)),norm(xA(:)-xA0(:))/norm(xA0(:)));
    
    tc_ = (1+sqrt(1+4*tc^2))/2;
    if ii>20 && fv(ii)<tol
        figure(10); clf; plot(fv);
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
    figure(10); clf; plot(fv);
    conv_t = toc(star_t);
    disp(['not converged within ' num2str(conv_t) ' seconds']);
end

% === sub-functions
function z = shrinkage(x, kappa)
z = max(0, 1 - kappa/norm(x))*x;

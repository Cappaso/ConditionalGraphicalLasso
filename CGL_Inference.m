function [ qY ] = CGL_Inference( X,B,A,Iter )
% Function to implement CGL Inference
% by using Mean-Field approach for Ising model.
% 
% output: qY, variational distribution.
% input: X n by (p-1)+1, where p-1 is feature dimensionality, 1 for bias parameter
%        B p by m, the model
%        A p by m*m, each column can be reshaped into a symmetric matrix, with diag(A) = 0,

% star_t = tic;
[n,p] = size(X);
m = size(B,2);

if nargin<=3; Iter = 10; end
maxIter = Iter*m;

% Y_ = -((-1).^Y); % {0,1} --> {-1,1}
X = full(X);

qY = zeros(n,m); % output probability for y = 1.
xi = zeros(n,m);

nu = X*B; % n by m
omega = X*A; % n by m*m

for ii = 1:maxIter
    tok = mod(ii,m); if tok==0; tok = m; end
    tok_c = [1:tok-1,tok+1:m]; % all index except 'tok'.
    
    shift = (tok-1)*m; % for omega
    qY(:,tok) = sigFun( 1.*(nu(:,tok) + sum(xi(:,tok_c).*omega(:,shift+tok_c), 2)) );
    xi(:,tok) = -1.*(1-qY(:,tok)) + qY(:,tok);
%     qY(:,tok) = sigFun( 1.*(nu(:,tok) + sum(qY(:,tok_c).*omega(:,shift+tok_c), 2)) );
    
end

% run_time = toc(star_t);
% disp(['CGL Inference used ' num2str(run_time) ' seconds']);

end

function y = sigFun(t)
y = 1./(1+exp(-t));
end


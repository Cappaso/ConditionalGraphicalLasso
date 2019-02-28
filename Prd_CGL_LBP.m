function Y = Prd_CGL_LBP(X,B,A)
n = size(X,1);
m = size(B,2);
Y = zeros(n,m);
print_cnt = 20;
print_step = ceil(n/print_cnt);

for in = 1:n
    Y(in,:) = map_maxprod_01( (X(in,:)*B)',reshape(X(in,:)*A, [m,m]) );
    
%     if mod(in,print_step)==0
%         fprintf('--Test data %d of %d\n',in,n)
%     end
end
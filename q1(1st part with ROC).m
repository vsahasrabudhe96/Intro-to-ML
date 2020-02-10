clear all
close all

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma(:,:,1) = [1,-0.9;-0.9,1]; Sigma(:,:,2) = [1,0.9;0.9,1];
p = [0.8,0.2];
label = rand(1,N)>=p(1);
Nc = [length(find(label==0)),length(find(label==1))];
N1 = Nc(1);
N2 = Nc(2);
x = zeros(n,N);
for l = 0:1
 x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'),
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'),
discriminantscore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));

for i = 1:40000
    h = i/4;
    gamma(i)= log(h); 
    
    decision = discriminantscore>=gamma(i);
    
    ind00 = find(decision==0 & label==0);
    p00(i) = length(ind00)/Nc(1); % probability of true negative
    
    ind01 = find(decision==0 & label==1); 
    p01(i) = length(ind01)/Nc(2); % probability of false negative
   
    ind10 = find(decision==1 & label==0); 
    p10(i) = length(ind10)/Nc(1); 

    ind11 = find(decision==1 & label==1);
    p11(i) = length(ind11)/Nc(2);
    perr(i) = ([p10(i),p01(i)]*Nc')/N;
end
k = max(p10);
j = max(p11);
[M,I] = min(perr(:))
% 
figure(3),
plot(p10,p11);hold on,
xlabel('False Positive');
ylabel('True Positive');
legend();
%plot(permin,'or');hold off;
plot(p10(I),p11(I),'-o');hold off;
axis([0 1 0 1]),

%%%%%%%% FISHER LDA %%%%%%%%
x1 = x(:,label==0);
x2 = x(:,label == 1);

mean1 = mean(x1,2);
mean2 = mean(x2,2);

var1 = cov(x1');
var2 = cov(x2');

Sb = (mean1 -mean2)*(mean1-mean2)';
Sw = var1+var2;

[V,D] = eig(inv(Sw),Sb);
[~,ind] = sort(diag(D),'descend');
W = V(:,ind(1));

prd1 = W'*x1;
prd2 =  W'*x2;


figure(4),
subplot(2,1,1), plot(x1(1,:),x1(2,:),'r*'); hold on;
plot(x2(1,:),x2(2,:),'bo'); axis equal, 
subplot(2,1,2), plot(prd1(1,:),zeros(1,N1),'r*'); hold on;
plot(prd2(1,:),zeros(1,N2),'bo'); axis equal,





function g = evalGaussian(x,mu,Sigma)
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
clear all
close all

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma(:,:,1) = [1,-0.9;-0.9,1]; Sigma(:,:,2) = [1,0.9;0.9,1];
sigma(:,:,1) = [1,0;0,1]; sigma(:,:,2) = [1,0;0,1];
p = [0.8,0.2];
label = rand(1,N)>=p(1);
Nc = [length(find(label==0)),length(find(label==1))];
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
discriminantscore = log(gaussiancalc(x,mu(:,2),sigma(:,:,2)))-log(gaussiancalc(x,mu(:,1),sigma(:,:,1)));

for i = 1:10000
    gamma(i)= log(i-1);     
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

figure(1), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold off;
axis equal,

figure(3),
plot(p10,p11);hold on,
xlabel('False Positive');
ylabel('True Positive');
legend();
title('ROC curve');
plot(p10(I),p11(I),'*');hold off;
axis([0 1 0 1]),

function g = gaussiancalc(x,mu,Sigma)
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2); 
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
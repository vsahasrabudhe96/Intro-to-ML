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
Sb = (mu(:,1) - mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1)+ Sigma(:,:,2);
[V,D] = eig(inv(Sw),Sb);
[~,ind] = sort(diag(D),'descend');
W = V(:,ind(1));
y = W'*x;
W = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*W; 
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; 

figure(3), clf,
plot(y(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(y(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
tau = [-N:N];
dS = (y >= 0);
for t = 1:20001
    dec = dS>=tau(t);
    lda00 = find(dec==0 & label==0);
    pr00(t) = length(lda00)/Nc(1); % probability of true negative
    lda01 = find(dec==0 & label==1); 
    pr01(t) = length(lda01)/Nc(2); % probability of false negative
    lda10 = find(dec==1 & label==0); 
    pr10(t) = length(lda10)/Nc(1); 
    lda11 = find(dec==1 & label==1);
    pr11(t) = length(lda11)/Nc(2);
    perrlda(t) = ([pr10(t),pr01(t)]*Nc')/N;
end
kl = max(pr10);
jl = max(pr11);
[Ml,Il] = min(perrlda(:))

figure(5),
plot(pr10,pr11);hold on,
xlabel('False Positive');
ylabel('True Positive');
legend();

plot(pr10(Il),pr11(Il),'*');hold off;
axis([0 1 0 1]),

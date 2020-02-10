
clear all, close all,

n = 2;      
N = 1000;   

% Class 0 parameters (2 gaussians)
mu(:,1) = [5;0]; mu(:,2) = [-5;7];
Sigma(:,:,1) = [5 2;2 6]; Sigma(:,:,2) = [3 2;2 4];
p0 = [0.5 0.5]; % probability split between 2 distributions

% Class 1 parameters (2 gaussians)
mu(:,3) = [0;1]; mu(:,4) = [4;6]; 
Sigma(:,:,3) = [3 2;2 8]; Sigma(:,:,4) = [5 1;1 9];
p1 = [0.5 0.5]; % probability split between 2 distributions

p = [0.3,0.7]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); 

% Draw samples from each class pdf
c0 = 0; 
c1=0; 
c2=0;
c3=0;
for i = 1:N
    if label(i) == 0
        % Class 0 samples for each gaussian based on their distribution
        dis = rand(1,1) > p0(1);
        if dis == 0
            x(:,i) = mvnrnd(mu(:,1),Sigma(:,:,1),1)';
            c0 = c0+1;
        else
            x(:,i) = mvnrnd(mu(:,2),Sigma(:,:,2),1)';
            c1 = c1+1;
        end
    end
    
    if label(i) == 1
        % Class 1 samples for each gaussian based on their distribution
        dis = rand(1,1) > p1(1);
        if dis == 0
            x(:,i) = mvnrnd(mu(:,3),Sigma(:,:,3),1)';
            c2 = c2+1;
        else
            x(:,i) = mvnrnd(mu(:,4),Sigma(:,:,4),1)';
            c3 = c3 + 1;
        end
    end
    
    
end

% Plot with class labels
figure(2), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2')

class1pdf = p1(1)*evalGaussian(x,mu(:,3),Sigma(:,:,3)) + p1(2)*evalGaussian(x,mu(:,4),Sigma(:,:,4));
class0pdf = p0(1)*evalGaussian(x,mu(:,1),Sigma(:,:,1)) + p0(2)*evalGaussian(x,mu(:,2),Sigma(:,:,2));
discriminantScore = log(class1pdf)-log(class0pdf);

for i= 1:10000
    gamma(i) =  log(i-1);
% compare score to threshold to make decisions
    decision = (discriminantScore >= gamma(i));

    ind00 = find(decision==0 & label==0); 
    p00(i) = length(ind00)/Nc(1); % probability of true negative
    ind10 = find(decision==1 & label==0); 
    p10(i) = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1); 
    p01(i) = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1); 
    p11(i) = length(ind11)/Nc(2); % probability of true positive
    perr(i) = ([p10(i),p01(i)]*Nc')/N;
end

[M,I] = min(perr(:));
figure(3),
xlabel('FPR'),ylabel('TPR');
plot(p10,p11);hold on;
plot(p10(I),p11(I),'*'),hold off;

figure(1), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),500);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),500);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,3),Sigma(:,:,3))+ evalGaussian([h(:)';v(:)'],mu(:,4),Sigma(:,:,4)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))+evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2))) - (gamma(N));
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,500,500);
figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 

legend(); 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

function g = evalGaussian(x,mu,Sigma)
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
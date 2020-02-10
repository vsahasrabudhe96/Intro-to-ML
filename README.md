Question 1 (60%)

The probability density function (pdf) for a 2-dimensional real-valued random vector X is as follows: p(x) = P(L = 0)p(xjL = 0) + P(L = 1)p(xjL = 1). Here L is the true class label that indicates which class-label-conditioned pdf generates the data.

The class priors are P(L = 0) = 0:8 and P(L = 1) = 0:2. The class class-conditional pdfs are p(xjL = 0) = g(xjm0; C0) and p(xjL = 1) = g(xjm1; C1), where g(xjm; C) is a multivariate Gaus-sian probability density function with mean vector m and covariance matrix C. The parameters of the class-conditional Gaussian pdfs are:

m0 = [ 00:1 ]  C0 = [ 10:9	10:9 ]  m1 = [00:1 ]  C1 = [01:9 01:9 ]

For numerical results requested below, generate 10000 samples according to this data distribu-tion, keep track of the true class labels for each sample. Save the data and use the same data set in all cases.

Minimum expected risk classification using the knowledge of true data pdf:

1. Specify the minimum expected risk classification rule in the form of a likelihood-ratio test:
?
> g, where the threshold g is a function of class priors and fixed (nonnegative) loss


values for each of the four cases D = ijL = j where D is the decision label that is either 0 or 1, like L.

2.	Implement this classifier and apply it on the 10K samples you generated. Vary the thresh-old g gradually from 0 to ¥ and for each value of the threshold compute the true posi-tive (detection) probability P(D = 1jL = 1) and the false positive (false alarm) probability P(D = 1jL = 0). Using these values, trace/plot an approximation of the ROC curve of the minimum expected risk classifier.

3.	Determine the threshold value that achieves minimum probability of error, and on the ROC curce, superimpose clearly (using a different color/shape marker) the true positive and false positive values attained by this minimum-P(error) classifier. Calculate and report an estimate of the minimum probability of error that is achievable for this data distribution.

Classification with incorrect knowledge of the data distribution (Naive Bayesian Classifier, which assumes features are independent given each class label): For the following items, assume that you know the true class prior probabilities and that you think the class conditional pdfs are both Gaussian with the true means, but (incorrectly) with covariance matrices both equal to the identity matrix. Analyze the impact of this model mismatch in this Naive Bayesian (NB) approach to classifier design.

1. Specify the minimum expected risk classification rule in the form of a likelihood-ratio test:
?


> g, where the class conditional pdfs are incorrectly known as specified in the naive Bayesian approximation above.


2.	Implement this naive-Bayesian classifier and apply it on the 10K samples you generated. Vary the threshold g gradually from 0 to ¥ and for each value of the threshold compute the true positive (detection) probability P(D = 1jL = 1) and the false positive (false alarm) probability P(D = 1jL = 0). Using these values, trace/plot an approximation of the ROC curve of the minimum expected risk decision rule.

3.	Determine the threshold value that achieves minimum probability of error, and on the ROC curve, superimpose clearly (using a different color/shape marker) the true positive and false positive values attained by this naive-Bayesian model based minimum-P(error) classifier. Calculate and report an estimate of the minimum probability of error that is achievable by the naive-Bayesian classification rule for this (true) data distribution.

In the third part of this exercise, repeat the same steps as in the previous two cases for the Fisher Linear Discriminant Analysis based classifier. Using the 10000 available samples, with sample average based estimates for mean and covariance matrix for each class, determine the Fisher LDA projection vector wLDA. For the classification rule wTLDAx compared to a threshold t, which takes values from ¥ to ¥, trace the ROC curve, identify the threshold at which the probability of error (based on sample count estimates) is minimized, and clearly mark that operating point on the ROC curve.

Note: In order for us to have a uniform solution across all submissions, When finding the Fisher LDA projection matrix, do not be concerned about the difference in the class priors. When determining the between-class and within-class scatter matrices, use equal weights for the class means and covariances, like we did in class.


Question 2 (30%)

For class labels 0 and 1, pick two class priors and two class-conditional pdfs (both in the form of mixtures of two Gaussians). Do not set the class priors to be equal. Use four different Gaussian components when constructing the class-conditional pdfs. Within each class-conditional pdf, do not select the mixture coefficients to be equal. Select your Gaussian mixtures to create an interesting/challenging example.

1.	Provide scatter plots of 1000 total samples from this data distribution. Note that both the true class label and within that class, which Gaussian component generates each sample should be selected randomly, in accordance with class prior probabilities, and Gaussian component probabilities (weights), respectively. Do NOT specify number of samples for any of the labels or components. Indicate true class label fo each sample with a different marker shape in the scatter plot.

2.	Determine the minimum-P(error) classification rule, specify it, draw its decision boundary superimposed on the scatter plot of data, and classify each sample with this classifier and with color cues indicate if the samples are correctly or incorrectly classifier. Using your data samples, calculate an estimate of the smallest probability of error achievable for this dataset and report it.


Question 3 (10%)

For a scalar random variable that may be generated by one of two classes, where class priors are equal and class-conditional pdfs are both unit-variance Gaussians with mean values -2 and 2, determine the classification rule that achieves minimum probability of error. Also express the smallest error probability achievable by this classifier in terms of definite integrals of Gaussian class conditional pdfs involved.

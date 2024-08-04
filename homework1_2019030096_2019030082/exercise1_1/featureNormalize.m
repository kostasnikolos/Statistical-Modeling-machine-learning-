function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


% ADD YOUR CODE

N = size(X, 1);
mu = (1/N) * sum(X, 1);
sigma = sqrt((1/(N-1)) * sum((X-ones(size(X)).*mu).^2, 1));
X_norm = (X - ones(size(X)).*mu) ./ sigma;


% ============================================================

end

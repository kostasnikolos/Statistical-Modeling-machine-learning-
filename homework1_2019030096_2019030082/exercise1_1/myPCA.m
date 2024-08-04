function [ eigenval, eigenvec, order] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [ eigenval, eigenvec, order] = mypca(X) computes eigenvectors of the autocorrelation matrix of X
%   Returns the eigenvectors, the eigenvalues (on diagonal) and the order 
%

% Useful values
[m, ~] = size(X);

% Make sure each feature from the data is zero mean
N = size(X, 1);
mu = (1/N) * sum(X, 1);
X_centered = X - ones(size(X)).*mu;
    

% ====================== YOUR CODE HERE ======================
%

Sigma = (1/m) * (X_centered' * X_centered);
[V,D] = eig(Sigma);
[D, ind] = sort(diag(D),1,'descend');
eigenval = D; % Eigenvalue
eigenvec = V(:,ind);  %Corresponding eigenvector
order = ind; %Sorting order



% =========================================================================

end

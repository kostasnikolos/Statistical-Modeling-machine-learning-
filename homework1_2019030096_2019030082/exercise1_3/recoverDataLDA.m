function [X_rec] = recoverDataLDA(Z, v)

% You need to return the following variables correctly.
X_rec = ones(size(Z, 1), length(v));

% ====================== YOUR CODE HERE ======================

X_rec = X_rec .* (v'/norm(v)) .* Z;

% =============================================================

end

function v=vech(X)
%This function computes the vech of a matrix
v=X(find(tril(ones(size(X)))));


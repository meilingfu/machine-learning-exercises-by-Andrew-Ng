function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

mid=X*theta;	
sig=sigmoid(mid);
J1=-sum(y.*log(sig)+(1-y).*log(1-sig))/m;
J2=lambda*sum(theta(2:end).^2)/(2*m);
J=J1+J2;

dif=sig-y;
grad(1)=X(:,1)'*dif/m;
for i=2:size(X,2)
    grad(i)=X(:,i)'*dif/m+lambda*theta(i)/m;
end


% =============================================================

end

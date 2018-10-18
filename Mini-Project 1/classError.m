function [ err ] = classError( y,yhat )
%CLASSERROR computes the class error for each class and averages it
%   
%   Input:
%       y:      the labels
%       yhat:   the output of the classifier
%
%   output:
%       err:    the class-averaged classification error
classes = unique(y);
err_ = zeros(1,length(classes));

for c=1:length(classes)
    err_(c) = sum((y~=yhat) & (y == classes(c)))./sum(y==classes(c));
end

err = mean(err_);

end


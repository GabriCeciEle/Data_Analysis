function [ErrorsArray] = arrayErrorsClass(trainingSet, testSet, trainingSet_labels, testSet_labels)

% INPUT: training/test sets with their respective labels. 
% OUTPUT: one matrix of errors using a uniform prior probability, where the number of row corresponds to 
% the model, and the number of column to the training/testing. Both are 
% 4x2 errors referring to different models.

%                     TRAIN   |  TEST
% diaglinear  -->    [        |     
% linear      -->             |
% diagquadratic-->            |
% (pseudo)quadratic-->        |       ]

% train a diaglinear classifier
[classifierDiaglinear_train,  ~, ~, errDiaglinear_train] = ...
    classification(trainingSet, trainingSet_labels, 'diaglinear', 'uniform');
yhatDiaglinear_test = predict(classifierDiaglinear_train, testSet);
errDiaglinear_test = classError(testSet_labels,yhatDiaglinear_test);

% train a linear classifier
[classifierLinear_train, ~, ~, errLinear_train] = ...
    classification(trainingSet, trainingSet_labels, 'linear', 'uniform');
yhatLinear_test = predict(classifierLinear_train, testSet);
errLinear_test = classError(testSet_labels,yhatLinear_test);

% train a diagquadratic classifier
[classifierDiagquadratic_train, ~, ~, errDiagquadratic_train] = ...
    classification(trainingSet, trainingSet_labels, 'diagquadratic', 'uniform');
yhatDiagquadratic_test = predict(classifierDiagquadratic_train, testSet);
errDiagquadratic_test = classError(testSet_labels,yhatDiagquadratic_test);

% train a quadratic classifier
[classifierQuadratic_train, ~, ~, errQuadratic_train] = ...
    classification(trainingSet, trainingSet_labels, 'pseudoquadratic', 'uniform');
yhatQuadratic_test = predict(classifierQuadratic_train, testSet);
errQuadratic_test = classError(testSet_labels,yhatQuadratic_test);


% array
ErrorsArray = [errDiaglinear_train, errDiaglinear_test; 
    errLinear_train, errLinear_test; 
    errDiagquadratic_train, errDiagquadratic_test; 
    errQuadratic_train, errQuadratic_test];


end


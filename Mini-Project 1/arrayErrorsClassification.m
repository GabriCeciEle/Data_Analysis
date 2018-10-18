function [ErrorsArray_empirical, ErrorsArray_prior] = arrayErrorsClassification(trainingSet, testSet, trainingSet_labels, testSet_labels)

% train a diaglinear classifier 'empirical'
[classifierDiaglinear_train,  ~, errDiaglinear_train_e] = classification(trainingSet, trainingSet_labels, 'diaglinear', 'empirical');
yhatDiaglinear_test = predict(classifierDiaglinear_train, testSet);
errDiaglinear_test_e = classificationError(testSet_labels,yhatDiaglinear_test);

% train a linear classifier 'empirical'
[classifierLinear_train, ~, errLinear_train_e] = classification(trainingSet, trainingSet_labels, 'linear', 'empirical');
yhatLinear_test = predict(classifierLinear_train, testSet);
errLinear_test_e = classificationError(testSet_labels,yhatLinear_test);

% train a diagquadratic classifier 'empirical'
[classifierDiagquadratic_train, ~, errDiagquadratic_train_e] = classification(trainingSet, trainingSet_labels, 'diagquadratic', 'empirical');
yhatDiagquadratic_test = predict(classifierDiagquadratic_train, testSet);
errDiagquadratic_test_e = classificationError(testSet_labels,yhatDiagquadratic_test);

% train a quadratic classifier 'empirical'
[classifierQuadratic_train, ~, errQuadratic_train_e] = classification(trainingSet, trainingSet_labels, 'pseudoquadratic', 'empirical');
yhatQuadratic_test = predict(classifierQuadratic_train, testSet);
errQuadratic_test_e = classificationError(testSet_labels,yhatQuadratic_test);

% array 'empirical'
ErrorsArray_empirical = [errDiaglinear_train_e, errDiaglinear_test_e; 
    errLinear_train_e, errLinear_test_e; 
    errDiagquadratic_train_e, errDiagquadratic_test_e; 
    errQuadratic_train_e, errQuadratic_test_e];


% train a diaglinear classifier
[classifierDiaglinear_train,  ~, errDiaglinear_train] = classification(trainingSet, trainingSet_labels, 'diaglinear', 'uniform');
yhatDiaglinear_test = predict(classifierDiaglinear_train, testSet);
errDiaglinear_test = classificationError(testSet_labels,yhatDiaglinear_test);

% train a linear classifier
[classifierLinear_train, ~, errLinear_train] = classification(trainingSet, trainingSet_labels, 'linear', 'uniform');
yhatLinear_test = predict(classifierLinear_train, testSet);
errLinear_test = classificationError(testSet_labels,yhatLinear_test);

% train a diagquadratic classifier
[classifierDiagquadratic_train, ~, errDiagquadratic_train] = classification(trainingSet, trainingSet_labels, 'diagquadratic', 'uniform');
yhatDiagquadratic_test = predict(classifierDiagquadratic_train, testSet);
errDiagquadratic_test = classificationError(testSet_labels,yhatDiagquadratic_test);

% train a quadratic classifier
[classifierQuadratic_train, ~, errQuadratic_train] = classification(trainingSet, trainingSet_labels, 'pseudoquadratic', 'uniform');
yhatQuadratic_test = predict(classifierQuadratic_train, testSet);
errQuadratic_test = classificationError(testSet_labels,yhatQuadratic_test);


% array
ErrorsArray_prior = [errDiaglinear_train, errDiaglinear_test; 
    errLinear_train, errLinear_test; 
    errDiagquadratic_train, errDiagquadratic_test; 
    errQuadratic_train, errQuadratic_test];
end

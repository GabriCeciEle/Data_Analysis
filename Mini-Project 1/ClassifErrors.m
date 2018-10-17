function ErrorsArray = ClassifErrors(trainingSet, testSet, trainingSet_labels, testSet_labels)

%% train a diaglinear classifier
[classifierDiaglinear_train, yhatDiaglinear_train, errDiaglinear_train] = classification(trainingSet, trainingSet_labels, 'diaglinear');
yhatDiaglinear_test = predict(classifierDiaglinear_train, testSet);
errDiaglinear_test = classificationError(testSet_labels,yhatDiaglinear_test);

%% train a linear classifier
[classifierLinear_train, yhatLinear_train, errLinear_train] = classification(trainingSet, trainingSet_labels, 'linear');
yhatLinear_test = predict(classifierLinear_train, testSet);
errLinear_test = classificationError(testSet_labels,yhatLinear_test);

%% train a diagquadratic classifier
[classifierDiagquadratic_train, yhatDiagquadratic_train, errDiagquadratic_train] = classification(trainingSet, trainingSet_labels, 'diagquadratic');
yhatDiagquadratic_test = predict(classifierDiagquadratic_train, testSet);
errDiagquadratic_test = classificationError(testSet_labels,yhatDiagquadratic_test);

%% train a quadratic classifier
[classifierQuadratic_train, yhatQuadratic_train, errQuadratic_train] = classification(trainingSet, trainingSet_labels, 'pseudoquadratic');
yhatQuadratic_test = predict(classifierQuadratic_train, testSet);
errQuadratic_test = classificationError(testSet_labels,yhatQuadratic_test);


%% array
ErrorsArray = [errDiaglinear_train, errDiaglinear_test; 
    errLinear_train, errLinear_test; 
    errDiagquadratic_train, errDiagquadratic_test; 
    errQuadratic_train, errQuadratic_test];
end


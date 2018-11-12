function [] = ModelBuilding(trainData,trainLabels,testData,classifiertype,bPcNumb)

stp = 10;
trainData_final = trainData(:,1:stp:end);
final_norm_train = zscore(trainData_final);  
[coeff,score,~,~,~] = pca(final_norm_train);
final_norm_test = (testData(:,1:stp:end) - mean(trainData_final,1))./std(trainData_final,0,1);
final_norm_score_test = final_norm_test*coeff;

[orderedInd, ~] = rankfeat(score, trainLabels, 'fisher');

[classifierKaggle, ~, ~,~] = classification(score(:,orderedInd(1:bPcNumb)),trainLabels,classifiertype,'uniform');
yhat_kaggle = predict(classifierKaggle,final_norm_score_test(:,orderedInd(1:bPcNumb)));
labelToCSV(yhat_kaggle,'labels_final.csv','csvlabels');


end
function [] = GuidesheetIV(trainData,trainLabels,testData)

[orderedInd, orderedPower] = rankfeat(trainData, trainLabels, 'fisher');

%standardized (z-score) data
%norm_data = zscore(trainData);
[coeff,score,latent,tsquared,variance] = pca(trainData(:, orderedInd(1:20)));

%% covariance matrix

cov_matrix_data = cov(trainData(:, orderedInd(1:20)));
variance_data=diag(cov_matrix_data)';

figure;
subplot(1,2,1);
imagesc(cov_matrix_data)%'InitialMagnification','fit');
colorbar
title('Before PCA');

cov_matrix_PCA = cov(score);
variance_data_PCA=diag(cov_matrix_PCA)';
subplot(1,2,2);
imagesc(cov_matrix_PCA, [0 1])%,'InitialMagnification',2000);
colorbar
title('After PCA');


%% (cumulated) explained variance
% cumul_variance = zeros(length(variance),1);
% cumul_variance(1) = variance(1);
% for i=2:length(variance)
%     cumul_variance(i) = cumul_variance(i-1) + variance(i);
% end

cumul_variance = cumsum(variance)/sum(variance);
figure
bar(cumul_variance);
title({'Cumulated explained variance in';'function of the principal components'});
xlabel('Number of principal components');
ylabel('Cumulated explained variance (%)');
%ylim([0 105]);
grid on;

% Sorting and plotting the loadings of PCA1 in the descending order
figure
bar(coeff(:,1))
title({'Loadings of the 1st principal component (unsorted)'});
xlabel('Variables');
ylabel('Loadings');
%xticks(1:28);
%xticklabels(varNames);
%xtickangle(90);

[sorted_coef1, sorting_protocol1] = sort(abs(coeff(:,1)),'descend');
%sorted_var_names1 = varNames(sorting_protocol1);
figure
bar(sorted_coef1);
% xticks(1:28);
% xticklabels(sorted_var_names1);
% xtickangle(90);
title({'Loadings of the 1st principal component (sorted)'});
xlabel('Variables');
ylabel('Loadings');



%% Cross validation for hyperparameters optimization

numMaxFolds = 10;
numMaxPCs = 100;

cpLabels = cvpartition(trainLabels,'kfold', numMaxFolds);

classif_error_train_Diaglin = [];
classif_error_test_Diaglin = [];
classif_error_train_LDA = [];
classif_error_test_LDA = [];
classif_error_train_Diagquad = [];
classif_error_test_Diagquad = [];
classif_error_train_QDA = [];
classif_error_test_QDA = [];

%(1:9:end)
for k=1:numMaxFolds    
    for p=1:numMaxPCs
        [training_set,test_set,training_labels,test_labels] = ...
        find_cvpartition(k, cpLabels, trainLabels, trainData(:,600:800));

        norm_train = zscore(training_set);
        
        [coeff,score,latent,tsquared,variance] = pca(norm_train);
        %[sorted_coef1, sorting_protocol1] = sort(abs(coeff(:,1)),'descend');        
        %cumul_variance = cumsum(variance)/sum(variance);
       
        norm_test = (test_set - mean(training_set,1))./std(training_set,0,1);
        %centered_test=zscore(testData(:,1:50:2048));
        norm_score_test = norm_test*coeff;
        
        [ErrorsArray,~] = ...
        arrayErrorsClassification(score(:,1:p), norm_score_test(:,1:p), training_labels, test_labels);
        
        classif_error_train_Diaglin(p,k) = ErrorsArray(1,1);
        classif_error_test_Diaglin(p,k) = ErrorsArray(1,2);
        classif_error_train_LDA(p,k) = ErrorsArray(2,1);
        classif_error_test_LDA(p,k) = ErrorsArray(2,2);
        classif_error_train_Diagquad(p,k) = ErrorsArray(3,1);
        classif_error_test_Diagquad(p,k) = ErrorsArray(3,2);
        classif_error_train_QDA(p,k) = ErrorsArray(4,1);
        classif_error_test_QDA(p,k) = ErrorsArray(4,2);
    end
end

TestErrorsCV = classif_error_test_Diaglin;
TestErrorsCV(:,:,2) = classif_error_test_LDA;
TestErrorsCV(:,:,3) = classif_error_test_Diagquad;
TestErrorsCV(:,:,4) = classif_error_test_QDA;

TrainErrorsCV = classif_error_train_Diaglin;
TrainErrorsCV(:,:,2) = classif_error_train_LDA;
TrainErrorsCV(:,:,3) = classif_error_train_Diagquad;
TrainErrorsCV(:,:,4) = classif_error_train_QDA;

meanTestErrorsCV = mean(TestErrorsCV,2);
meanTrainErrorsCV = mean(TrainErrorsCV,2);

% min = 1;
% 
% for p=1:size(meanTestErrorsCV,1)
%     for m=1:size(meanTestErrorsCV,3)
%         if meanTestErrorsCV(p,1,m)< min
%             Results.bestPCsNumber = p;
%             Results.classifierType = m;
%             min=meanTestErrorsCV(p,1,m);
%         end
%     end
% end

figure('name','test')
plot(meanTestErrorsCV(:,1,1))
hold on
plot(meanTestErrorsCV(:,1,2))
hold on
plot(meanTestErrorsCV(:,1,3))
hold on
plot(meanTestErrorsCV(:,1,4))
legend('diaglin','LDA','diagquad','QDA')

figure('name','train')
plot(meanTrainErrorsCV(:,1,1))
hold on
plot(meanTrainErrorsCV(:,1,2))
hold on
plot(meanTrainErrorsCV(:,1,3))
hold on
plot(meanTrainErrorsCV(:,1,4))
legend('diaglin','LDA','diagquad','QDA')

figure('name','comparison')
subplot(2,2,1)
plot(meanTestErrorsCV(:,1,1))
hold on
plot(meanTrainErrorsCV(:,1,1))
legend('test','train')
title('diaglin')
subplot(2,2,2)
plot(meanTestErrorsCV(:,1,2))
hold on
plot(meanTrainErrorsCV(:,1,2))
legend('test','train')
title('LDA')
subplot(2,2,3)
plot(meanTestErrorsCV(:,1,3))
hold on
plot(meanTrainErrorsCV(:,1,3))
legend('test','train')
title('diagquad')
subplot(2,2,4)
plot(meanTestErrorsCV(:,1,4))
hold on
plot(meanTrainErrorsCV(:,1,4))
legend('test','train')
title('QDA')


%% Forward features selection

% Application to the whole dataset
classifiertype = 'diaglinear';

fun = @(xT,yT,xt,yt)length(yt)*(classificationError(yt,predict(fitcdiscr(xT,yT,'discrimtype',classifiertype),xt)));

opt = statset('Display','iter','MaxIter',100);
cp = cvpartition(trainLabels,'kfold',10);

[selectedFeatures,history] = sequentialfs(fun,trainData(:,600:800),trainLabels,'nfeatures',2,'cv',cp);


% Cross validation

numMaxFolds = 10;
numMaxFeatures = 10;
opt = statset('Display','iter','MaxIter',100);
classifiertype={'diaglinear','linear','diagquadratic','pseudoquadratic'};
classif_error_test = [];
selectedFeatures = [];

for ct=1:4
    fun = @(xT,yT,xt,yt)length(yt)*...
            (classificationError(yt,predict(fitcdiscr(xT,yT,'discrimtype',classifiertype{ct}),xt)));

    cp = cvpartition(trainLabels,'kfold',numMaxFolds);
    [selFeatures,history] = sequentialfs(fun,trainData(:,600:800),trainLabels,'cv',cp,'options',opt);
    
    classif_error_test(1,ct) = history.Crit(end);   
    classif_error_test(2,ct) = length(find(selFeatures==1));
end       


%% Model building

train = zscore(trainData(:,500:1000));
[coeff,score] = pca(train);

fn = 10;

test = (testData - mean(trainData,1))./std(trainData,0,1);
test = test(:,500:1000);
test = test*coeff;

[classifierKaggle, ~, ~,~] = classification(score(:,fn),trainLabels,'diaglinear','empirical');
yhat_kaggle = predict(classifierKaggle,test(:,fn));
labelToCSV(yhat_kaggle,'labels_3.csv','csvlabels');


end


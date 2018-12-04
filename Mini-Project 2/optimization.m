function [Results] = optimization(Alpha, lambda, train, validation, trainData, validationData)
%% Elastic Nets
Results.ElasticNets.nonZeros = [];
Results.ElasticNets.MSE_training = [];

for alpha_= 1:length(Alpha)
    [B,FitInf] = lasso(trainData,train,'Lambda',lambda,'Alpha',Alpha(alpha_));
    Regressed = validationData*B + FitInf.Intercept;
    
    for lambda_= 1:size(lambda,2)
        Results.ElasticNets.validationErr(alpha_,lambda_) = immse(validation,Regressed(:,lambda_));
    end
    
    Results.ElasticNets.nonZeros = [Results.ElasticNets.nonZeros;FitInf.DF];
    Results.ElasticNets.MSE_training = [Results.ElasticNets.MSE_training;FitInf.MSE];
    
    [M_opt(1,alpha_),I_opt] = min(Results.ElasticNets.validationErr(alpha_,:));
    lambda_opt(1,alpha_) = lambda(I_opt);
end
    
[Results.ElasticNets.validationErr,a_opt] = min(M_opt);
Results.ElasticNets.optimalAlpha = Alpha(a_opt);
Results.ElasticNets.optimalLambda = lambda_opt(a_opt);

%% PCA + Regression
[coeff,trainDatascore, ~, ~,explained] = pca(trainData);
validationDatascore = validationData*coeff;

Results.PCAandRegression.cumulative = cumsum(explained)/sum(explained);
Results.PCAandRegression.index = find(Results.PCAandRegression.cumulative>0.9,1);

I_train = ones(size(train,1),1);
I_validation = ones(size(validation,1),1);

trainErr = [];
trainErr_second = [];
validationErr = [];
validationErr_second = [];

for ind = 1:150%size(trainDatascore,2)
    
    FM_train = trainDatascore(:,1:ind);
    FM_validation = validationDatascore(:,1:ind);
    Train = [I_train FM_train];
    Validation = [I_validation FM_validation];
    Train_second = [I_train FM_train FM_train.^2];
    Validation_second = [I_validation FM_validation FM_validation.^2];

    b = regress(train,Train);
    b_second = regress(train,Train_second);
    
    trainErr = [trainErr, immse(train,Train*b)];
    trainErr_second = [trainErr_second,immse(train,Train_second*b_second)];
    
    validationErr = [validationErr, immse(validation,Validation*b)];
    validationErr_second = [validationErr_second,immse(validation,Validation_second*b_second)];
    
end

Results.PCAandRegression.validationErr = [validationErr;validationErr_second];
[Results.PCAandRegression.validationErr_opt, Results.PCAandRegression.numPCs_opt]= min(validationErr);
[Results.PCAandRegression.validationErr_optSecond, Results.PCAandRegression.numPCs_optSecond]= min(validationErr_second);

end


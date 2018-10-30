function [] = GuidesheetIV(trainData,trainLabels,testData)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%standardized (z-score) data
%norm_data = zscore(trainData);
[coeff,score,latent,tsquared,variance] = pca(trainData);

%% covariance matrix

cov_matrix_data = cov(trainData);
variance_data=diag(cov_matrix_data)';

figure;
subplot(1,2,1);
imagesc(cov_matrix_data)%'InitialMagnification','fit');
colorbar
title('Before PCA');

cov_matrix_PCA = cov(score);
variance_data_PCA=diag(cov_matrix_PCA)';
subplot(1,2,2);
imagesc(cov_matrix_PCA)%,'InitialMagnification',2000);
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

end


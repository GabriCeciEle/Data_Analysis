clear all
close all
clc

%% Data Import 
load('trainSet.mat');
load('testSet.mat');
load('trainLabels.mat');

%% Division
classA = [];
classB = [];

for sample_ = 1:size(trainData,1)
    if trainLabels(sample_) == 0
        classA = [classA; trainData(sample_, :)];
    else
        classB = [classB; trainData(sample_,:)];
    end
end


%% Histogram
figure('name','Features distribution')
for subplotNumber = 1:10
    subplot(2,5,subplotNumber)
    histogram(classA(:,715+subplotNumber),[0:0.1:1])
    hold on
    histogram(classB(:,715+subplotNumber),[0:0.1:1])
    xlabel('Amplitude [\muV]')
    ylabel('Number of samples')
    title(int2str(715+subplotNumber))
end

figure('name','Features distribution')
for subplotNumber = 1:10
    subplot(2,5,subplotNumber)
    boxplot(classA(:,715+subplotNumber))
    hold on
    boxplot(classB(:,715+subplotNumber))
    title(int2str(715+subplotNumber))
end
%legend('classA','classB')

%% Boxplot
figure('name','2 Features')
subplot(1,2,1)
boxplot(classA(:,716))
hold on
boxplot(classB(:,716))
title('Difference')
subplot(1,2,2)
boxplot(classA(:,723))
hold on
boxplot(classB(:,723))
title('Similarity')


%% Boxplot
figure('name','2 Features')
subplot(1,2,1)
boxplot(classA(:,716),'Notch','on')
hold on
boxplot(classB(:,716),'Notch','on')
title('Difference')
subplot(1,2,2)
boxplot(classA(:,723),'Notch','on')
hold on
boxplot(classB(:,723),'Notch','on')
title('Similarity')






clear all
close all
clc

%% Data Import 
load('trainSet.mat');
load('testSet.mat');
load('trainLabels.mat');

%% Division
error = [];
correct = [];

for sample_ = 1:size(trainData,1)
    if trainLabels(sample_) == 0
        correct = [correct; trainData(sample_, :)];
    else
        error = [error; trainData(sample_,:)];
    end
end

figure('name','Random Correct');
for i=1:10:456
     plot(correct(i,:),'-')%'MarkerSize',8,'MarkerFaceColor','red');
     hold on
end
title('Random')
xlabel('Time [ms]','Fontsize',10,'Color','k');
ylabel('Amplitude [\muV]','Fontsize',10,'Color','k');

figure('name','Random Error');
for i=1:10:141
     plot(error(i,:),'-')%'MarkerSize',8,'MarkerFaceColor','red');
     hold on
end
title('Random')
xlabel('Time [ms]','Fontsize',10,'Color','k');
ylabel('Amplitude [\muV]','Fontsize',10,'Color','k');


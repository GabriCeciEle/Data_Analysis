clear all
close all
clc

%% Data Import 

load('spikes2018.mat');
size = size(spikes);
SizeAnswer=['There are: ' int2str(size(1)) ' spikes and ' int2str(size(2)) ' time steps']

%% 2D plot of 10 random spikes

Spikeidx = [4 750 1560 2320 3100 3850 4200 4680 5300 5999];
figure('name','10 random spikes')
for j = Spikeidx
    plot(spikes(j,:));
    hold on;
end
title('Spikes (10 random out of 6000)')
xlabel('Time [ms]','Fontsize',10,'Color','k');
ylabel('Amplitude [\muV]','Fontsize',10,'Color','k');

%% 2D plot of 60 spikes
figure(2);
for i=1:100:6000
     plot(spikes(i,:),'-')%'MarkerSize',8,'MarkerFaceColor','red');
     hold on;
end

% %% 2D plot of ALL spikes
% 
% figure(3);
% for i=1:6000
%      plot(spikes(i,:),'-')%'MarkerSize',8,'MarkerFaceColor','red');
%      hold on;
% end

%% PCA analysis to define features

[coeff,score,~,~,explained] = pca(spikes');
spikesPCA = coeff(:,:);

%% plot 2 features of the spike

figure('name','2 features of the spike')
%scatter(score(sub_indexes,1),score(sub_indexes,2),10,'filled');

%% Visualisation of histogram and boxplots of the three features 

figure('name','Features distributions');

subplot(2,3,1);
histogram(spikesPCA(:,1));
ylabel('Number of samples','Fontsize',12,'Color','k');
title('Feature #1')

subplot(2,3,2);
histogram(spikesPCA(:,2));
ylabel('Number of samples','Fontsize',12,'Color','k');
title('Feature #2')

subplot(2,3,3);
histogram(spikesPCA(:,3));
ylabel('Number of samples','Fontsize',12,'Color','k');
title('Feature #3')

subplot(2,3,4);
boxplot(spikesPCA(:,1));
grid on;
ylabel('Number of samples','Fontsize',12,'Color','k');

subplot(2,3,5);
boxplot(spikesPCA(:,2));
grid on;
ylabel('Number of samples','Fontsize',12,'Color','k');

subplot(2,3,6);
boxplot(spikesPCA(:,3));
grid on;
ylabel('Number of samples','Fontsize',12,'Color','k');

%% plot matrix

figure('name','PlotMatrix');
plotmatrix(spikes(:,1:10:end))


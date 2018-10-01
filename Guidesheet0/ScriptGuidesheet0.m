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
%% Scatter
figure
scatter(spikes(:,50),spikes(:,60));

%% Histograms
figure('name','Features distributions')
subplot(2,3,1)
histogram(spikes(:,40))
grid on
xlabel('Amplitude','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('40sec')

subplot(2,3,2)
histogram(spikes(:,45))
grid on
xlabel('Amplitude','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('45sec')

subplot(2,3,3)
histogram(spikes(:,50))
grid on
xlabel('Amplitude','Fontsize',12,'Color','k');
ylabel('Number of samples','Fontsize',12,'Color','k');
title('50sec')

subplot(2,3,4)
histogram(spikes(:,55))
grid on
xlabel('Amplitude','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('55sec')

subplot(2,3,5)
histogram(spikes(:,60))
grid on
xlabel('Amplitude','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('60sec')

subplot(2,3,6)
histogram(spikes(:,65))
grid on
xlabel('Amplitude','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('65sec')

%% Boxplots
figure('name','Features distributions')
subplot(2,3,1)
boxplot(spikes(:,40))
grid on
ylabel('Amplitude','Fontsize',12,'Color','k')
title('40sec')

subplot(2,3,2)
boxplot(spikes(:,45))
grid on
ylabel('Amplitude','Fontsize',12,'Color','k')
title('45sec')

subplot(2,3,3)
boxplot(spikes(:,50))
grid on
ylabel('Amplitude','Fontsize',12,'Color','k')
title('50sec')

subplot(2,3,4)
boxplot(spikes(:,55))
grid on
ylabel('Amplitude','Fontsize',12,'Color','k')
title('55sec')

subplot(2,3,5)
boxplot(spikes(:,60))
grid on
ylabel('Amplitude','Fontsize',12,'Color','k')
title('60sec')

subplot(2,3,6)
boxplot(spikes(:,65))
grid on
ylabel('Amplitude','Fontsize',12,'Color','k')
title('65sec')

%% plot matrix
figure('name','PlotMatrix');
plotmatrix(spikes(:,1:10:end))

%% 3D visualization
figure()
scatter3(spikes(:,61),spikes(:,71),spikes(:,81))

%% kmeans
selectedFeatures = [61 71 81];
idx=kmeans(spikes(:,selectedFeatures),3);
figure
gplotmatrix(spikes(:,selectedFeatures),idx);




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
figure('name','Random Spikes');
for i=1:100:6000
     plot(spikes(i,:),'-')%'MarkerSize',8,'MarkerFaceColor','red');
     hold on
end
title('Random Spikes')
xlabel('Time [ms]','Fontsize',10,'Color','k');
ylabel('Amplitude [\muV]','Fontsize',10,'Color','k');

%% Scatter of 2 features 
figure('name','2 features')
scatter(spikes(:,50),spikes(:,60));
title('2 features')
xlabel('Amplitude [\muV]','Fontsize',10,'Color','k');
ylabel('Amplitude [\muV]','Fontsize',10,'Color','k');

%% Histograms
figure('name','Features distributions')
subplot(2,3,1)
histogram(spikes(:,40))
grid on
xlabel('Amplitude [\muV]','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('40sec')

subplot(2,3,2)
histogram(spikes(:,45))
grid on
xlabel('Amplitude [\muV]','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('45sec')

subplot(2,3,3)
histogram(spikes(:,50))
grid on
xlabel('Amplitude [\muV]','Fontsize',12,'Color','k');
ylabel('Number of samples','Fontsize',12,'Color','k');
title('50sec')

subplot(2,3,4)
histogram(spikes(:,55))
grid on
xlabel('Amplitude [\muV]','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('55sec')

subplot(2,3,5)
histogram(spikes(:,60))
grid on
xlabel('Amplitude [\muV]','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('60sec')

subplot(2,3,6)
histogram(spikes(:,65))
grid on
xlabel('Amplitude [\muV]','Fontsize',12,'Color','k')
ylabel('Number of samples','Fontsize',12,'Color','k')
title('65sec')

%% Boxplots
figure('name','Features distributions')
subplot(2,3,1)
boxplot(spikes(:,40))
grid on
ylabel('Amplitude [\muV]','Fontsize',12,'Color','k')
title('40sec')

subplot(2,3,2)
boxplot(spikes(:,45))
grid on
ylabel('Amplitude [\muV]','Fontsize',12,'Color','k')
title('45sec')

subplot(2,3,3)
boxplot(spikes(:,50))
grid on
ylabel('Amplitude [\muV]','Fontsize',12,'Color','k')
title('50sec')

subplot(2,3,4)
boxplot(spikes(:,55))
grid on
ylabel('Amplitude [\muV]','Fontsize',12,'Color','k')
title('55sec')

subplot(2,3,5)
boxplot(spikes(:,60))
grid on
ylabel('Amplitude [\muV]','Fontsize',12,'Color','k')
title('60sec')

subplot(2,3,6)
boxplot(spikes(:,65))
grid on
ylabel('Amplitude [\muV]','Fontsize',12,'Color','k')
title('65sec')

%% plot matrix
figure('name','PlotMatrix');
plotmatrix(spikes(:,1:10:end))

%% 3D visualization
figure('name','3 features')
scatter3(spikes(:,61),spikes(:,71),spikes(:,81))

%% kmeans with k=2
k=2;
selectedFeatures = [61 71 81];
[idx, C2, sum2]=kmeans(spikes(:,selectedFeatures),k);
figure('name','gplotmatrix with 2 clusters')
gplotmatrix(spikes(:,selectedFeatures),[],idx);

%% mean
neuron1 = [];
neuron2 = [];

for spike_=1:6000
    if idx(spike_)==1
        neuron1 = [neuron1; spikes(spike_,:)];
    else
        neuron2 = [neuron2; spikes(spike_,:)];
    end
end

neuron1 = mean(neuron1);
neuron2 = mean(neuron2);

figure('name','2 different neurons')
plot(neuron1)
grid on
hold on
plot(neuron2)
title('2 different neurons')
xlabel('Time [s]')
ylabel('Amplitude [\muV]')
legend('Neuron 1','Neuron 2')

%% kmeans with k=3
k=3;
selectedFeatures = [61 71 81];
[idx, C3, sum3]=kmeans(spikes(:,selectedFeatures),k);
figure('name','gplotmatrix with 3 clusters')
gplotmatrix(spikes(:,selectedFeatures),[],idx);

%% mean
neuron1 = [];
neuron2 = [];
neuron3 = [];

for spike_=1:6000
    if idx(spike_)==1
        neuron1 = [neuron1; spikes(spike_,:)];
    else
        if idx(spike_)==2
        neuron2 = [neuron2; spikes(spike_,:)];
        else 
        neuron3 = [neuron3; spikes(spike_,:)];
        end
    end
end

neuron1 = mean(neuron1);
neuron2 = mean(neuron2);
neuron3 = mean(neuron3);

figure('name','3 different neurons')
plot(neuron1)
grid on
hold on
plot(neuron2)
hold on
plot(neuron3)
title('3 different neurons')
xlabel('Time [s]')
ylabel('Amplitude [\muV]')
legend('Neuron 1','Neuron 2','Neuron 3')

%% Squared error
SquaredErrorAnswer=['With 2 clusters we obtain ' int2str(sum2(1)) ' and ' int2str(sum2(2)) ', with 3 clusters ' int2str(sum3(1)) ' and ' int2str(sum3(2)) ' and ' int2str(sum3(3)) ' .']

%% Evaluating clusters 
% EVA = evalclusters(X, CLUST, CRITERION)
% CRITERION is a string representing the
    % criterion to be used. The value of CRITERION could be 'CalinskiHarabasz',
    % 'Silhouette', 'gap' or 'DaviesBouldin'.
EVA_CalinskiHarabasz = evalclusters(spikes,idx,'CalinskiHarabasz');
EVA_Silhouette = evalclusters(spikes,idx,'Silhouette');


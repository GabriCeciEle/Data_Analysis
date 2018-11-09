clc
close all
clear all

labels = [0 0 0 0 1 1 0 0 1 0];
predicted = [0 1 1 0 0 1 0 1 1 0];
classError = classError(labels,predicted);
classificationError = classificationError(labels',predicted'); 
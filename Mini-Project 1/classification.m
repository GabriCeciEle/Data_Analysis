function [classifier, yhat_train, err] = classification(dataset, labels, classifierType)

classifier = fitcdiscr(dataset, labels, 'discrimtype', classifierType);
yhat_train = predict(classifier, dataset);
err = classificationError(labels, yhat_train);

end


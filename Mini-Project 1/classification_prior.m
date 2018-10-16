function [classifier, yhat_train, err] = classification_prior (dataset, labels, classifier_type, priortype)

classifier= fitcdiscr(dataset, labels, 'discrimtype', classifier_type, 'prior', priortype);
yhat_train = predict(classifier, dataset);
err = classification_error(labels, yhat_train);

end

function [classifier, yhat_train, classificerr, classerr] = classification(dataset, labels, classifier_type, priortype)

classifier= fitcdiscr(dataset, labels,'discrimtype', classifier_type,'prior', priortype);
yhat_train = predict(classifier, dataset);

classificerr = classificationError(labels, yhat_train);
classerr = classError(labels, yhat_train);

end

function [training_set, test_set, training_labels, test_labels] = find_cvpartition(k, partition, labels, features)

% take the features for the samples selected to be in the train set.
% INPUT: number of folds (k), the partition class already created, the
% array of labels, and the dataset. 
% OUTPUT: This function extracts, for a given
% fold k, the training and test sets, and training and test labels.
% WARNING: this function is designed to work only with a labels vector
% containing two classes, "0" and "1".

training_set = features(find(partition.training(k)==1),:); 
test_set = features(find(partition.test(k)==1),:); 
training_labels = labels(find(partition.training(k)==1));
test_labels = labels(find(partition.test(k)==1));

end


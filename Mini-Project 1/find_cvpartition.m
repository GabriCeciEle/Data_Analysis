function [training_set, test_set, training_labels, test_labels] = find_cvpartition(k, partition, labels, features)
% take the features for the samples selected to be in the train set

training_set = features(find(partition.training(k)==1),:); 
test_set = features(find(partition.test(k)==1),:); 
training_labels = labels(find(partition.training(k)==1));
test_labels = labels(find(partition.test(k)==1));

end


function [classif_error] = classificationError(labels,newLabels)

classif_error = sum(labels~=newLabels)/size(labels,1);
end


function [class_error] = classError(labels, predictedLabels, weightA, weightB)
totA = 0;
totB = 0;
misclassifiedA = 0;
misclassifiedB = 0;

for sample_ = 1:length(labels)
    if labels(sample_) == 0
        totA = totA+1;
    else
        totB = totB+1;
    end
end

for sample_=1:length(labels)
    if labels(sample_) == 0 && labels(sample_)~=predictedLabels(sample_)
        misclassifiedA = misclassifiedA+1;
    end
    if labels(sample_) == 1 && labels(sample_)~=predictedLabels(sample_)
        misclassifiedB = misclassifiedB+1;
    end
end

class_error = weightA*(misclassifiedA/totA) + weightB*(misclassifiedB/totB);

end


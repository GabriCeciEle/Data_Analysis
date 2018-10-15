function [class_error] = classError(weightA, weightB, labels, predicted_labels)


correctA = 0;
correctB = 0;


for i = 1:size(classA,1)
    if(labels(i)==trainLabels(i))
        correctA = correctA + 1;
    end
end


class_error = weightA*((size(classA,1)-correctA)/size(classA,1)) + weightB*((size(classB,1)-correctB)/size(classB,1));



end


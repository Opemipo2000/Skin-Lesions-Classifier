% This code was developed by student F219244 on 20/02/23
% This code finds various performance metric of classifying benign and
% malignant skin lesions via border feature extraction from the mask of the images.

% Uploading the dataset including the images and the correct label 
% (malignant or benign) for each image

image_dataset = imageDatastore("lesionimages/", "FileExtensions", ".jpg");

masks_dataset = imageDatastore("masks/", "FileExtensions", ".png");

images = readall(image_dataset); % read all images 

masks = readall(masks_dataset); % read all masks 

details = readlines("groundtruth.txt");

labels = extractAfter(details,",");

labels = labels(1:end-1);

% Train the SVM Model with the mask of the images and labels and get the confusion
% matrix

%extract the border features from the images

for i = 1:length(masks)
   stats{i} = regionprops(masks{i}, "circularity"); 
   stats{i} = stats{1,i}(end);
   image_matrix(i) = stats{1,i}(:);
end

image_matrix = cell2mat(struct2cell(image_matrix));
image_matrix = reshape(image_matrix,[200,1]);

% perform classification using 10CV

rng(1); % let's all use the same seed for the random number generator
svm = fitcsvm(image_matrix, labels);
cvsvm = crossval(svm);
pred = kfoldPredict(cvsvm);
[cm, order] = confusionmat(labels, pred); % cm is the confusion 

[accuracy, sensitivity, specificity, precision] = metric_performance(cm);

function [accuracy, sensitivity, specificity, precision] = metric_performance(confusion_matrix)
% This function was inspired by Preetham Manjunatha from the link: 
% https://uk.mathworks.com/matlabcentral/fileexchange/105825-multiclass-metrics-of-a-confusion-matrix

% Define the parts of the confusion matrix

    TP = confusion_matrix(1,1);
    FP = confusion_matrix(2,1);
    FN = confusion_matrix(1,2);
    TN = confusion_matrix(2,2);
    
    % Calculate the metrics with the above values
    accuracy = (TP+TN) / (TP+TN+FP+FN);
    sensitivity = TP / (TP + FN);
    precision = TP / (TP + FP);
    
    if ((TN/(TN+FP)) > 1)
        specificity = 1;
    elseif ((TN/(TN+FP)) < 0)
        specificity = 0;
    else
        specificity = TN/(TN+FP);
    end
end

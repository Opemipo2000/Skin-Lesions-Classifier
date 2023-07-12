% This code was developed by student F219244 on 16/02/23
% This code finds various performance metric of classifying benign and
% malignant skin lesions with data preprocessing via changing original images to grayscale.

% To the grayscaled images the image mask is applied to make all the backgrounds a constant balck colour.

image_dataset = imageDatastore("lesionimages/", "FileExtensions", ".jpg");

masks_dataset = imageDatastore("masks/", "FileExtensions", ".png");

images = readall(image_dataset); % read all images 

masks = readall(masks_dataset); % read all masks 

details = readlines("groundtruth.txt");

labels = extractAfter(details,",");

labels = labels(1:end-1);

% convert all images to grayscale 

for i = 1:length(images)
    images{i} = im2gray(images{i});
end

% Train the SVM Model with the images and labels and get the confusion
% matrix

for i = 1:length(images)
    masked_images{i} = imfuse(images{i},masks{i}, 'blend');
    ch = imhist(masked_images{i});
    allhists(i,:) = ch(:);
end

imshow(masked_images{1});

% do PCA on data
% the k used here was 2
[pcs evals projdata] = mypca(allhists);
images_matrix = projdata(:,1:20);

% perform classification using 10CV

rng(1); % let's all use the same seed for the random number generator
svm = fitcsvm(images_matrix, labels);
cvsvm = crossval(svm);
pred = kfoldPredict(cvsvm);
[cm, order] = confusionmat(labels, pred); % cm is the confusion 

[accuracy, sensitivity, specificity, precision] = metric_performance(cm);

function [pcs, evals, projdata] = mypca(data)
% Referenced from Lab 6 solutions.
    c = cov(data); % covariance matrix
    [v, d] = eig(c); % get eigenvectors
    d = diag(d);
    [~, ind] = sort(d, 'descend'); % sort eigenvalues
    pcs = v(:,ind);
    evals = d(ind);
    projdata = data * pcs; % project onto PCA space
end

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

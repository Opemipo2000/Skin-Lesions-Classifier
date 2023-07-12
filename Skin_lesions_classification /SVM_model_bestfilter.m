% This code was developed by student F219244 on 16/02/23
% This code finds various performance metric of classifying benign and
% malignant skin lesions via PCA with data preprocessing via filtered images from the original image.

% Uploading the dataset including the images and the correct label 
% (malignant or benign) for each image

image_dataset = imageDatastore("lesionimages/", "FileExtensions", ".jpg");

images = readall(image_dataset); % read all images 

% applying mean filter to each image

mean_filter = fspecial('average', 3);

for i = 1:length(images)
    images{i} = imfilter(images{i}, mean_filter);
end

details = readlines("groundtruth.txt");

labels = extractAfter(details,",");

labels = labels(1:end-1);

% Train the SVM Model with the images and labels and get the confusion
% matrix

for i = 1:length(images)
    ch = colourhist(images{i});
    allhists(i,:) = ch(:);
end

% do PCA on data
%images_matrix = projdata(:,1:k); 
% the k used here was the best value that gives the highest accuracy without overfitting.
[pcs evals projdata] = mypca(allhists);
images_matrix = projdata(:,1:20);

% perform classification using 10CV

rng(1); % let's all use the same seed for the random number generator
svm = fitcsvm(images_matrix, labels);
cvsvm = crossval(svm);
pred = kfoldPredict(cvsvm);
[cm, order] = confusionmat(labels, pred); % cm is the confusion 

[accuracy, sensitivity, specificity, precision] = metric_performance(cm);

function H = colourhist(image)
% Referenced from Lab 4 solutions.
% generate 8x8x8 RGB colour histogram from image
    noBins = 8; % 8 bins (along each dimension)
    binWidth = 256 / noBins; % width of each bin
    H = zeros(noBins, noBins, noBins); % empty histogram to start with

    [n, m, d] = size(image);
    data = reshape(image, n*m, d); % reshape image into 2-d matrix with one row per pixel

    ind = floor(double(data) / binWidth) + 1; % calculate into which bin each pixel falls

    for i=1:length(ind)
        H(ind(i,1), ind(i,2), ind(i,3)) = H(ind(i,1), ind(i,2), ind(i,3)) + 1; % increment bin
    end
    H = H / sum(sum(sum(H))); % normalise histogram
end

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

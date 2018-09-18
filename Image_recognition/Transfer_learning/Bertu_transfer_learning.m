clear all
% Retrain AlexNet to recognise Bertu.
% retrained network thinks everything is Bertu...
% "loss" is too low
imds = imageDatastore('Bertu_repository', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% resize for AlexNet
inputSize = [227 227];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);

% usage of large image dataset. Sort label out.

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized'); % split...
% images into 70% training 30% validation

%show some images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

net = alexnet; % load AlexNet
net.Layers %show network: 5 convolutional layers, 3 fully connected layers
inputSize = net.Layers(1).InputSize % 227 X 227 X rgb

% replace final 3 layer for retraining
layersTransfer = net.Layers(1:end-3);% last three layers are designed for...
% 1000 classification classes, extract all but last three layers
numClasses = numel(categories(imdsTrain.Labels));

numClasses = 1;

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

%augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
%     'DataAugmentation',imageAugmenter);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);


augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',3, ... % 6 before
    'InitialLearnRate',1e-4, ... %1e-4
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,layers,options);

% moment of truth on validation images
[YPred,scores] = classify(netTransfer,augimdsValidation);

% plot four randoms
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

% accuracy
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

%% try picture outside of 

I = imread('control.jpg');
I_red = imresize(I,[227,227]);

image(I_red)
[label, prob] = classify(netTransfer, I_red); % identify object in image
thresh = 0.5;
if max(prob)>=thresh
    title(['Object identified: ' char(label)]);
else
    title(['Below threshold (P = ' num2str(round(max(prob),2)) ')'])
end
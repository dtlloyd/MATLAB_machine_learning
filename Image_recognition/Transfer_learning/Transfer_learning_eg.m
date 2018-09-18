% Transfer learning example
% https://uk.mathworks.com/help/nnet/examples/transfer-learning-using-alexnet.html
% try  openExample('nnet/FeatureExtractionUsingAlexNetExample')
% in the command line to get Merchdata.zip
unzip('MerchData.zip');
imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); % imagedatastore for efficient memory...
% usage of large image dataset. Sort label out.

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized'); % split...
% images into 70% training 30% validation

%show some images
% numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end

net = alexnet; % load AlexNet
net.Layers %show network: 5 convolutional layers, 3 fully connected layers
inputSize = net.Layers(1).InputSize % 227 X 227 X rgb

% replace final 3 layer for retraining
layersTransfer = net.Layers(1:end-3);% last three layers are designed for...
% 1000 classification classes, extract all but last three layers
numClasses = numel(categories(imdsTrain.Labels))

numClasses = 5

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% "Transfer the layers to the new classification task by replacing the last 
% three layers with a fully connected layer, a softmax layer, and a 
% classification output layer. Specify the options of the new fully connected 
% layer according to the new data. Set the fully connected layer to have the 
% same size as the number of classes in the new data. To learn faster in the 
% new layers than in the transferred layers, increase the WeightLearnRateFactor 
% and BiasLearnRateFactor values of the fully connected layer"

% Data augmentation: resize, flip and shift to avoid overfitting to
% precise image composition
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter); 

% leave validation images alone, except for resizing
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Training options
% wish for faster learning rate in new layers, slower in older layers
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
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


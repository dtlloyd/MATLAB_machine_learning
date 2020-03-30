% needs Neural Network Toolbox
% and AlexNet pre-trained network:
% https://uk.mathworks.com/matlabcentral/fileexchange/59133-neural-network-toolbox-tm--model-for-alexnet-network
% and Webcam support package:
% https://uk.mathworks.com/matlabcentral/fileexchange/45182-matlab-support-package-for-usb-webcams

% Grab webcam image
% run AlexNet ANN to check if there is a cat present
% make a (celebratory) sound if there is a cat

%camList = webcamlist;

camera = webcam(1); % open camera instance

nnet = alexnet; % import pre-trained ANN
% google net requires upgrading
fig = figure(1);

tb = uicontrol(fig, 'Style', 'togglebutton', 'String', 'Stop'); % create...
% stop button

thresh = 0.1 % propbability threshold for sucessfull detection

while tb.Value == 0 % exits loop when button is pressed
    
    drawnow
    picture = camera.snapshot; % grab webcam image
    picture = imresize(picture,[227,227]); % AlexNet likes 227 x 227
    
    image(picture) % show web cam image
    
    [label, prob] = classify(nnet, picture); % identify object in image
    
    # if classification probability above threshold print label
    if max(prob)>=thresh
        title(['Object identified: ' char(label)]);
    else
        title(['Below threshold (P = ' num2str(round(max(prob),2)) ')'])
    end
    daspect([1 1 1])
    set(gca,'xtick',[],'ytick',[])
    
    % if there is a cat in the picture play a pleasant sound
    if contains(char(label),'cat')
        sound(S(2).y,S(2).Fs)
    end

end

clear camera

%% run AlexNet on static imported image
clear all
S(2) = load('handel');

nnet = alexnet;

I = imread('Price_cat.jpg');
I_red = imresize(I,[227,227]);

image_in = I_red;
image(image_in)
daspect([1 1 1])

label = classify(nnet, image_in);
title(['Object identified: ' char(label)]);
daspect([1 1 1])
set(gca,'xtick',[],'ytick',[])

if contains(char(label),'cat')
    sound(S(2).y,S(2).Fs)
end
%% AlexNet cat classes
%  281: 'tabby, tabby cat',
%  282: 'tiger cat',
%  283: 'Persian cat',
%  284: 'Siamese cat, Siamese',
% 285: 'Egyptian cat',

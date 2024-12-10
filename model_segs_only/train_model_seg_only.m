% train_model_better.m
% 
% Description:
% Alternate training method using just the images with segmentations
% This method is faster than using all of them.
%
% Metrics:
% Date/Time           : 2024-12-10 15:24:13
% Author information  : Tim Perr (tlperr@mtu.edu)
%                     : Ryan Moreau (rjmoreau@mtu.edu)
%                     : Matt Plansinis (mvplansi@mtu.edu)
% Source code license : GPL v3 (https://www.gnu.org/licenses/gpl-3.0.txt)
% OS                  : Windows 11
% RAM                 : 15 GB
% CPU model           : 12th Gen Intel(R) Core(TM) i7-12700 @ 2.10GHz
% CPU/Core count      : 12
% Software/Language   : MATLAB
% Version             : R2022a
% Pre-req/Dependency  : Image Processing Toolbox, Deep Learning Toolbox
% Compilation command : NA
% Compilation time    : NA
% Execution command   : train_model_better (in command window)
%                     : matlab -nodisplay -nosplash -nodesktop -r \
%                           "run('train_model_better.m'); exit;"
% Execution time      : Varies based off network input size and the number
%                       of input images

% clearing everything
clear
close all
imtool close all

% gather relevant files
model_inputs = "model_inputs_better\";
model_inputs_files = dir(fullfile(model_inputs, '**', '*.dcm'));
model_inputs_file_paths = fullfile({model_inputs_files.folder}, {model_inputs_files.name}); % Full paths

labels_base = "labels_better\";
label_files = dir(fullfile(labels_base, '**', '*.png'));
label_file_paths = fullfile({label_files.folder}, {label_files.name}); % Full paths

% removing errors
load("errors_list.mat", "errlist");
for e = 1:size(errlist, 1)
    err = errlist(e, :);
    model_inputs_file_paths = model_inputs_file_paths(~contains(model_inputs_file_paths, err));
    label_file_paths = label_file_paths(~contains(label_file_paths, err));
end

% load labels
classes = ["background", "cancer"];
labelIDs = [0, 1];
pxds = pixelLabelDatastore(label_file_paths, classes, labelIDs);
pxds.ReadFcn = @(x) imresize(imread(x), [512 512]); % resizing

% load input images
imds = imageDatastore(model_inputs_file_paths, "FileExtensions",".dcm");
imds.ReadFcn = @(x) imresize(dicomread(x), [512 512]); %resizing

% combine
ds = pixelLabelImageDatastore(imds, pxds, "OutputSize", [512, 512]);

% setup for training
numClasses = numel(classes);
lgraph = unetLayers([512, 512], numClasses);
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ... 
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', ds, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% train network
net = trainNetwork(ds, lgraph, options);
% save so you don't have to train again
save("trained_network.mat", "net")
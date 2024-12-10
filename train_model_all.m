% train_model_all.m
% 
% Description:
% Training model using all input images
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
% Execution command   : train_model_all (in command window)
%                     : matlab -nodisplay -nosplash -nodesktop -r \
%                           "run('train_model_all.m'); exit;"
% Execution time      : Varies based off network input size and the number
%                       of input images

% clear stuff
clear
close all
imtool close all

% loading in all relevant images
src_neg_assess_base = "CPTAC-PDA_SourceImages_NegativeAssessments\manifest-1688579611120\CPTAC-PDA";
neg_files = dir(fullfile(src_neg_assess_base, '**', '*.dcm')); 
neg_file_paths = fullfile({neg_files.folder}, {neg_files.name}); % Full paths

src_seg_seed_base = "CPTAC-PDA_SourceImages_SEGSandSeedpoints\manifest-1688579331162\CPTAC-PDA";
seg_seed_files = dir(fullfile(src_seg_seed_base, '**', '*.dcm'));
seg_seed_file_paths = fullfile({seg_seed_files.folder}, {seg_seed_files.name}); % Full paths

labels_base = "Z:\CompVision\Fall2024\Proj5\labels";
label_files = dir(fullfile(labels_base, '**', '*.png'));
label_file_paths = fullfile({label_files.folder}, {label_files.name}); % Full paths

% removing errors
load("errors_list.mat", "errlist");
for e = 1:size(errlist, 1)
    err = errlist(e, :);
    neg_file_paths = neg_file_paths(~contains(neg_file_paths, err));
    seg_seed_file_paths = seg_seed_file_paths(~contains(seg_seed_file_paths, err));
    label_file_paths = label_file_paths(~contains(label_file_paths, err));
end

% loading in labels
classes = ["background", "cancer"];
labelIDs = [0, 1];
pxds = pixelLabelDatastore(label_file_paths, classes, labelIDs);
pxds.ReadFcn = @(x) imresize(imread(x), [128 128]); % resizing

% loading in inputs
imds = imageDatastore([seg_seed_file_paths, neg_file_paths], "FileExtensions",".dcm");
imds.ReadFcn = @(x) imresize(dicomread(x), [128 128]); %resizing

% combining inputs with labels
ds = pixelLabelImageDatastore(imds, pxds, "OutputSize", [128, 128]);

% model setup
numClasses = numel(classes);
lgraph = unetLayers([128, 128], numClasses);
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ... 
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', ds, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% train and save network
net = trainNetwork(ds, lgraph, options);
save("trained_network_all.mat", "net")
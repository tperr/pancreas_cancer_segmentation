% testing_stuff.m
% 
% Description:
% Testing ground to see if trained model does anything useful
% Loads model from trained_network(some extension).mat, picks some
% arbitrary patient, arbitrary contour, and displays input image and output
% segmentation to visually check if its right
% Spoiler alert: its not right
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
% Execution command   : testing_stuff (in command window)
%                     : matlab -nodisplay -nosplash -nodesktop -r \
%                           "run('testing_stuff.m'); exit;"
% Execution time      : Varies based off nw size

% clearing everything
clear
close all
imtool close all

% loading network variable (net) and defining paths
load("trained_network.mat")
src_neg_assess_base = "CPTAC-PDA_SourceImages_NegativeAssessments\manifest-1688579611120\CPTAC-PDA\";
src_seg_seed_base = "CPTAC-PDA_SourceImages_SEGSandSeedpoints\manifest-1688579331162\CPTAC-PDA\";
tumor_annotation_base = "CPTAC-PDA_Tumor-Annotations\manifest-1688576409969\CPTAC-PDA\";
labels_base = "labels\";

% read annotation to find one with contour
patient_dir = "C3L-00189";
% paths for patient
patient_src_dir = "CPTAC-PDA_SourceImages_SEGSandSeedpoints\manifest-1688579331162\CPTAC-PDA\C3L-00189\07-28-2003-NA-CT ABDOMEN  PELVIS ENHANCEDAB-63347\303.000000-AP 52.5MM 20 ASIR-64593\";
patient_annotation_dir = "CPTAC-PDA_Tumor-Annotations\manifest-1688576409969\CPTAC-PDA\C3L-00189\07-28-2003-NA-CT ABDOMEN  PELVIS ENHANCEDAB-63347\303.000000-Pre-dose  PANCREATIC DUCT - 1-113.4\1-1.dcm";
patient_seed_dir = "CPTAC-PDA_Tumor-Annotations\manifest-1688576409969\CPTAC-PDA\C3L-00189\07-28-2003-NA-CT ABDOMEN  PELVIS ENHANCEDAB-63347\303.000000-Pre-dose  PANCREATIC DUCT - 1 - SEED POINT-066.4\1-1.dcm";

% volume/spatial data, annotation/seed point data
center = [512 512]/2;
[vol, spat] = dicomreadVolume(patient_src_dir);
annotations_info = dicominfo(patient_annotation_dir);
annotations_contour = dicomContours(annotations_info);

seed_info = dicominfo(patient_seed_dir);
seed_contour = dicomContours(seed_info);
seed_point = seed_contour.ROIs.ContourData{1}{1};

% getting annotated mask and overlay
segs = annotations_contour.ROIs.ContourData{1,1};
picked_contour = segs{1};
x = picked_contour(:,1);
y = picked_contour(:,2);
z = picked_contour(1,3);
sliceIndex = find(spat.PatientPositions(:, 3) == z);
spacings = spat.PixelSpacings(sliceIndex, :);
x_overlay = round(((x - seed_point(:,1)) + center(1)) / spacings(1));
y_overlay = round(((y - seed_point(:,2)) + center(2)) / spacings(2));

% reading in input image and displaying what the mask should be over it
cur_img = dicomread(fullfile(patient_src_dir, "1-" + num2str(sliceIndex, "%03d") + ".dcm"));
figure("Name", "Outline of seg")
imshow(cur_img, []);
hold on
plot(x_overlay, y_overlay, "b.", "LineWidth",5);
hold off

% predicting what the mask should be
inputSize = net.Layers(1).InputSize(1:2);
cur_img = imresize(cur_img, inputSize);
pred_mask = semanticseg(cur_img, net, 'ExecutionEnvironment', 'auto');

% turning into logical, just takes the cancer areas
pred_mask = pred_mask == "cancer";
figure("Name", "Predicted mask")
imshowpair(pred_mask, cur_img, 'blend');

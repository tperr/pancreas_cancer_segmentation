% stuff.m
% 
% Description:
% Testing ground to see how stuff works, informal and messy
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
% Execution command   : stuff (in command window)
%                     : matlab -nodisplay -nosplash -nodesktop -r \
%                           "run('stuff.m'); exit;"
% Execution time      : Varies based off network input size and the number
%                       of input images

clear
close all
imtool close all

src_neg_assess_base = "CPTAC-PDA_SourceImages_NegativeAssessments\manifest-1688579611120\CPTAC-PDA\";
src_seg_seed_base = "CPTAC-PDA_SourceImages_SEGSandSeedpoints\manifest-1688579331162\CPTAC-PDA\";
tumor_annotation_base = "CPTAC-PDA_Tumor-Annotations\manifest-1688576409969\CPTAC-PDA\";
labels_base = "labels\";

% patient
patient_dir = "C3L-02888";
patient_seg_base = fullfile(src_seg_seed_base, patient_dir); % base/patient
patient_seg_base_path = dir(patient_seg_base);
patient_seg_base_path = patient_seg_base_path(3).name;
patient_seg_base = fullfile(patient_seg_base, patient_seg_base_path); % patient/scan
patient_seg_base_path = dir(patient_seg_base); % could contain n > 1 scans, plan code accordingly
patient_seg_base_path = patient_seg_base_path(3).name; 
src_seg_dir = fullfile(patient_seg_base, patient_seg_base_path);

patient_annot_base = fullfile(tumor_annotation_base, patient_dir); % base/patient
patient_annot_base_path = dir(patient_annot_base);
patient_annot_base_path = patient_annot_base_path(3).name;
annotationos_dir = fullfile(patient_annot_base, patient_annot_base_path); % patient/base of annotations

patient_mask_dir = fullfile(labels_base, patient_dir);

segseed_dirs = dir(annotationos_dir);
segseed_dirs = segseed_dirs(3:end);
% single point
seed_path = fullfile(annotationos_dir, segseed_dirs(1).name, "\1-1.dcm");
% segmentation
seg_path = fullfile(annotationos_dir, segseed_dirs(2).name, "\1-1.dcm");

annotations_info = dicominfo(seg_path);
annotations_contour = dicomContours(annotations_info);
seed_info = dicominfo(seed_path);
seed_contour = dicomContours(seed_info);
seed_point = seed_contour.ROIs.ContourData{1}{1};

num_imgs = length(dir(fullfile(src_seg_dir, "*.dcm")));
segs = annotations_contour.ROIs.ContourData{1,1};
slice = round(length(segs)/2);
img_num = abs(round(segs{slice}(1, 3)));
img_num = abs(round(seed_point(3)));

%img_num = num_imgs - img_num;

xycoords = segs{slice}(:, 1:2);

cur_img = dicomread(fullfile(src_seg_dir, "1-" + num2str(img_num, "%03d") + ".dcm"));

center = size(cur_img)/2;

figure("Name", "Contour")
pcont = plotContour(annotations_contour);

% volume data
[vol, spat] = dicomreadVolume(src_seg_dir);

[rows, cols, ~, numSlices] = size(vol);
maskVolume = zeros(rows, cols, numSlices, 'logical');
contour_data = annotations_contour.ROIs.ContourData{1}; 
%center = [0 0];
for i = 1:numel(contour_data)
    if isempty(contour_data{i})
        continue; % should never happen
    end
    
    % Extract x, y, z coordinates
    % Nx3 matrix: [x, y, z]
    % x and y in mm, z is slice
    points = contour_data{i}; 
    x = points(:,1);
    y = points(:,2);
    z = points(1,3);
    sliceIndex = find(spat.PatientPositions(:, 3) == z);
    spacings = spat.PixelSpacings(sliceIndex, :);

    x_overlay = round(((x - seed_point(:,1)) + center(1)) / spacings(1));
    y_overlay = round(((y - seed_point(:,2)) + center(2)) / spacings(2));
    % Rasterize the contour into a binary mask for the slice
    polyMask = poly2mask(x_overlay, y_overlay, rows, cols);
    
    % Insert the binary mask into the 3D mask volume
    maskVolume(:,:,sliceIndex) = maskVolume(:,:,sliceIndex) | polyMask;
end

img_num = sliceIndex; % just show last slice
cur_img = dicomread(fullfile(src_seg_dir, "1-" + num2str(img_num, "%03d") + ".dcm"));

figure("Name","Mask overlayed")
imshowpair(maskVolume(:,:,sliceIndex), cur_img, 'blend');

figure("Name","Mask outlined")
imshow(cur_img, []);
hold on
plot(x_overlay, y_overlay, 'b.', "LineWidth", 3); % just annotation
hold off
return

errs = []
for a = 1:numel(patients_anno)-2
    n = patients_anno(a+2).name
    disp(n)
    if (contains([patients_segd.name], n))
        errs = [errs; n]
    end
    disp(contains([patients_segd.name], n))
end
errs

%img_collection_annotations = dicomCollection(src_seg_dir);
% this opens up source scans
%dicomBrowser(img_collection_annotations)

disp("patient processed")
classes = ["background", "pancreas"];
labelIDs = [0, 1];
pxds = pixelLabelDatastore(patient_mask_dir, classes, labelIDs);
imds = imageDatastore(src_seg_dir, "FileExtensions",".dcm");
imds.ReadFcn = @(x) dicomread(x);

ds = pixelLabelImageDatastore(imds, pxds, "OutputSize", [512, 512]);
disp("initial loading")
numClasses = numel(classes);
lgraph = unetLayers([512, 512], numClasses);
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ... % make bigish
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', ds, ...
    'Plots', 'training-progress', ...
    'Verbose', true);
disp("nw ready")
net = trainNetwork(ds, lgraph, options);
disp("nw trained")
% preprocessing_better.m
% 
% Description:
% Preprocessing for model that uses only inputs with segmentations
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
% Execution command   : preprocessing_better (in command window)
%                     : matlab -nodisplay -nosplash -nodesktop -r \
%                           "run('preprocessing_better.m'); exit;"
% Execution time      : Varies based off size of dataset

% clears stuff
clear
close all
fclose("all");
imtool close all

% for error logging
fid = fopen('errors.txt', 'w');

% define base paths
src_neg_assess_base = "CPTAC-PDA_SourceImages_NegativeAssessments\manifest-1688579611120\CPTAC-PDA\";
src_seg_seed_base = "CPTAC-PDA_SourceImages_SEGSandSeedpoints\manifest-1688579331162\CPTAC-PDA\";
tumor_annotation_base = "CPTAC-PDA_Tumor-Annotations\manifest-1688576409969\CPTAC-PDA\";
labels_base = "labels_better\";
inputs_base = "model_inputs_better\";

% keeping track of stuff with errors to handle them later or never
errlist = [];

% gen masks for all patients with annotations
% patients
disp("annotations")
% go for all patients that have annotations (SEGSandSeedpoints)
patients_anno = dir(src_seg_seed_base);
for patient_name = 3:numel(patients_anno)
    % extract mask, patient, and patient input dirs
    patient_dir = patients_anno(patient_name).name;
    patient_mask_dir = fullfile(labels_base, patient_dir);
    patient_input_dir = fullfile(inputs_base, patient_dir);
    disp(patient_dir)

    % go for each of the "days" that a patient may have
    patient_seg_base = fullfile(src_seg_seed_base, patient_dir);
    patient_seg_base_path = dir(patient_seg_base);
    for day = 3:numel(patient_seg_base_path)
        patient_day_path_name = patient_seg_base_path(day).name; % subfolder
        patient_day_path_full = fullfile(patient_seg_base, patient_day_path_name); % patient/scan
        patient_day_path = dir(patient_day_path_full); 
        % there are some non-standard structures, using try ignores them,
        % goes through each scan for each day
        try
            for scan = 3:numel(patient_day_path)
                % full source and annotation dirs
                scan_name = patient_day_path(scan).name;
                src_seg_dir = fullfile(patient_day_path_full, scan_name);
                
                annotationos_dir = fullfile(tumor_annotation_base, patient_dir, patient_day_path_name); % patient/base of annotations
                if exist(annotationos_dir, "dir")
                    % get segmentations and seed points
                    segseed_dirs = dir(annotationos_dir);
                    segseed_dirs = segseed_dirs(3:end);
                    % single point
                    seed_path = fullfile(annotationos_dir, segseed_dirs(scan - 2).name, "\1-1.dcm");
                    % segmentation
                    seg_path = fullfile(annotationos_dir, segseed_dirs(scan - 1).name, "\1-1.dcm");
                    
                    annotations_info = dicominfo(seg_path);
                    annotations_contour = dicomContours(annotations_info);
                    
                    seed_info = dicominfo(seed_path);
                    seed_contour = dicomContours(seed_info);
                    seed_point = seed_contour.ROIs.ContourData{1}{1};
                    % all images are 512x512, should probably be changed at some point
                    center = [512, 512]/2; 
                    
                    % volume data
                    [vol, spat] = dicomreadVolume(src_seg_dir);
                    [rows, cols, ~, numSlices] = size(vol); 

                    % init mask and get contours
                    maskVolume = zeros(rows, cols, numSlices, 'logical');
                    contour_data = annotations_contour.ROIs.ContourData{1}; 
                    if numel(contour_data) > 0
                        mkdir(patient_mask_dir + "\" + scan_name);
                        mkdir(patient_input_dir + "\" + scan_name);
                    end
    
                    % for each contour do processing
                    for i = 1:numel(contour_data)
                        if isempty(contour_data{i})
                            continue; % should never happen
                        end
                        
                        % get coords for contour
                        points = contour_data{i}; 
                        x = points(:,1);
                        y = points(:,2);
                        z = points(1,3);
                        sliceIndex = find(spat.PatientPositions(:, 3) == z);
                        spacings = spat.PixelSpacings(sliceIndex, :);
                    
                        % stupid axial plane
                        x_overlay = round(((x - seed_point(:,1)) + center(1)) / spacings(1));
                        y_overlay = round(((y - seed_point(:,2)) + center(2)) / spacings(2));
                        polyMask = poly2mask(x_overlay, y_overlay, rows, cols);
                        
                        % recalc mask if needed and saves
                        maskVolume(:,:,sliceIndex) = maskVolume(:,:,sliceIndex) | polyMask;
                        maskFile = sprintf('%s/%s/mask_slice_%d.png', patient_mask_dir, scan_name, sliceIndex);
                        imwrite(maskVolume(:,:,sliceIndex), maskFile);
                        
                        % bc no admin, i have to re-save source images not sym link it
                        file_regex = "^1-0*" + sliceIndex + "\.dcm$";
                        potential_files = dir(fullfile(src_seg_dir, "*.dcm"));
                        matchingFiles = potential_files(~cellfun('isempty', regexp({potential_files.name}, file_regex)));

                        cur_img = dicomread(fullfile(src_seg_dir, {matchingFiles.name}));
                        inputFile = sprintf('%s/%s/input_slice_%d.dcm', patient_input_dir, scan_name, sliceIndex);
                        dicomwrite(cur_img, inputFile)
                    end
                end
            end
            disp(patient_dir)
        catch ME
            % saves error info so i can ignore for now
            fprintf(fid, '%s: %s\n %s\n', patient_dir, ME.identifier, ME.message);
            for s=1:numel(ME.stack)
                fprintf(fid, '\t%s at %s\n\tFile: %s\n', ME.stack(s).file, ME.stack(s).line, ME.stack(s).name);
            end
            fprintf(fid, '\n');
            errlist = [errlist; patient_dir];
        end
    end
end

% wrapup
fclose(fid);
save("errors_list.mat", "errlist");
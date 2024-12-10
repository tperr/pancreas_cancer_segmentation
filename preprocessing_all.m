% preprocessing_all.m
% 
% Description:
% Preprocessing for model that uses all images for segmentations
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
% Execution command   : preprocessing_all (in command window)
%                     : matlab -nodisplay -nosplash -nodesktop -r \
%                           "run('preprocessing_all.m'); exit;"
% Execution time      : Varies based off size of dataset

% clear all
clear
close all
fclose("all");
imtool close all

% for error logging
fid = fopen('errors.txt', 'w');

% paths
src_neg_assess_base = "CPTAC-PDA_SourceImages_NegativeAssessments\manifest-1688579611120\CPTAC-PDA";
src_seg_seed_base = "CPTAC-PDA_SourceImages_SEGSandSeedpoints\manifest-1688579331162\CPTAC-PDA";
tumor_annotation_base = "CPTAC-PDA_Tumor-Annotations\manifest-1688576409969\CPTAC-PDA";
labels_base = "Z:\CompVision\Fall2024\Proj5\labels";

errlist = [];

% gen masks for all patients with annotations
% patients
disp("annotations")
% do for all patients w/annotations
patients_anno = dir(src_seg_seed_base);
for patient_name = 3:numel(patients_anno)
    % get patient annotation dir, and mask dir
    patient_dir = patients_anno(patient_name).name;
    patient_mask_dir = fullfile(labels_base, patient_dir);
    
    disp(patient_dir)
    
    % get "days" and iterate through them
    patient_seg_base = fullfile(src_seg_seed_base, patient_dir); % base/patient
    patient_seg_base_path = dir(patient_seg_base);
    for day = 3:numel(patient_seg_base_path)
        patient_day_path = patient_seg_base_path(day).name; % subfolder
        patient_day_path_full = fullfile(patient_seg_base, patient_day_path); % patient/scan
        patient_day_path = dir(patient_day_path_full); % could contain n > 1 scans, plan code accordingly
        
        % err checking, it happens
        try
            for scan = 3:numel(patient_day_path)
                % full source and annotation dirs for each scan per "day"
                scan_name = patient_day_path(scan).name;
                src_seg_dir = fullfile(patient_day_path_full, scan_name);
                
                patient_annot_base = fullfile(tumor_annotation_base, patient_dir); % base/patient
                patient_annot_base_path = dir(patient_annot_base);
                patient_annot_base_path = patient_annot_base_path(day).name;
                annotationos_dir = fullfile(patient_annot_base, patient_annot_base_path); % patient/base of annotations
                
                % segmentations and seeds
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
                
                center = [512, 512]/2; % all images are 512x512
                
                % volume data
                [vol, spat] = dicomreadVolume(src_seg_dir);
                
                % mask and contour data
                [rows, cols, ~, numSlices] = size(vol); 
                maskVolume = zeros(rows, cols, numSlices, 'logical');
                contour_data = annotations_contour.ROIs.ContourData{1}; 
                
                % go through each contour
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
                    sliceIndex = find(spat.PatientPositions(:, 3) == z); % + seed_point(1, 3);
                    spacings = spat.PixelSpacings(sliceIndex, :);
                
                    % stupid axial plane
                    x_overlay = round(((x - seed_point(:,1)) + center(1)) / spacings(1));
                    y_overlay = round(((y - seed_point(:,2)) + center(2)) / spacings(2));
                    polyMask = poly2mask(x_overlay, y_overlay, rows, cols);
                    
                    % Insert the binary mask into the 3D mask volume
                    maskVolume(:,:,sliceIndex) = maskVolume(:,:,sliceIndex) | polyMask;
                end
                % save entire mask for patient
                mkdir(patient_mask_dir + "\" + scan_name);
                for i = 1:numSlices % i believe numSlices should be the same for each scan, so this should be okay
                    maskFile = sprintf('%s/%s/mask_slice_%d.png', patient_mask_dir, scan_name, i);
                    imwrite(maskVolume(:,:,i), maskFile);
                end
            end
            disp(patient_dir)
        % err logging
        catch ME
            fprintf(fid, '%s: %s\n %s\n', patient_dir, ME.identifier, ME.message);
            for s=1:numel(ME.stack)
                fprintf(fid, '\t%s at %s\n\tFile: %s\n', ME.stack(s).file, ME.stack(s).line, ME.stack(s).name);
            end
            fprintf(fid, '\n');
            errlist = [errlist; patient_dir];
        end
    end
end

% gen masks for all patients without annotations
% patients
disp("No annotations")
% for each without annotations
patients_anno = dir(src_neg_assess_base);
for patient_name = 3:numel(patients_anno)
    % mask and patient dir
    patient_dir = patients_anno(patient_name).name;
    patient_mask_dir = fullfile(labels_base, patient_dir);
    
    disp(patient_dir)

    % for each day
    patient_neg_base = fullfile(src_neg_assess_base, patient_dir); % base/patient
    patient_neg_base_path = dir(patient_neg_base);
    for day = 3:numel(patient_neg_base_path)
        patient_day_path = patient_neg_base_path(day).name; % subfolder, always only 1 and is useless
        patient_day_path_full = fullfile(patient_neg_base, patient_day_path); % patient/scan
        patient_day_path = dir(patient_day_path_full); % could contain n > 1 scans, plan code accordingly
        
        % for each scan write an empty image
        for scan = 3:numel(patient_day_path)
            scan_name = patient_day_path(scan).name;
            numSlices = numel(dir(fullfile(patient_day_path_full, scan_name))) - 2;
        
            mkdir(patient_mask_dir + "\" + scan_name);
            for i = 1:numSlices 
                maskFile = sprintf('%s/%s/mask_slice_%d.png', patient_mask_dir, scan_name, i);
                imwrite(zeros(512, 512), maskFile);
            end
        end
    end
    disp(patient_dir)

end

% wrapup
fclose(fid);
save("errors_list.mat", "errlist");
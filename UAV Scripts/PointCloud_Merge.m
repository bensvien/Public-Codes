%%Matlab code to merge las files and output as a single las.
%By Dr. Ben Vien
%Version 1.0
%Update: 9/01/2025


%% 02: Define the output LAS file path
%%
outputFile = 'merged_filedummy.las'; 

%% 01: las filename
%Define the input LAS file paths
%
lasFile1 = 'cloud2469e92110b5f28b_Block_0.las';  % Path to the first LAS file
lasFile2 = 'cloud2469e92110b5f28b_Block_1.las';  % Path to the second LAS file
% datetime('now') %Timecheck

%% Read the LAS files
tic
reader1 = lasFileReader(lasFile1);
reader2 = lasFileReader(lasFile2);
pc1 = readPointCloud(reader1);
pc2 = readPointCloud(reader2);
toc

% Merge the point cloud data
mergedLocations = [pc1.Location; pc2.Location];
mergedColors = [pc1.Color; pc2.Color];
mergedIntensity=[pc1.Intensity; pc2.Intensity];

%%% If require all attributes (Uncomment)
%mergedAttributes = struct();
%attributeFields = fieldnames(pc1);
%for i = 1:numel(attributeFields)
%    fieldName = attributeFields{i};
%    mergedAttributes.(fieldName) = [pc1.(fieldName); pc2.(fieldName)];
%end

%%
% Create a merged point cloud
mergedPointCloud = pointCloud(mergedLocations, 'Color', mergedColors,'Intensity',mergedIntensity);
disp('Compiled')
%%
% Write the merged point cloud to a new LAS file
lasWriter=lasFileWriter(outputFile);
writePointCloud(lasWriter,mergedPointCloud);
disp(['Merged LAS file!', outputFile]);
datetime('now')

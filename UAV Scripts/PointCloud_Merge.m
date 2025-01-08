%%Matlab code to merge las files 
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
toc

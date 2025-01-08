% 
%Define the input LAS file paths
%25W October 2024
lasFile1 = 'cloud2469e92110b5f28b_Block_0.las';  % Path to the first LAS file
lasFile2 = 'cloud2469e92110b5f28b_Block_1.las';  % Path to the second LAS file
% datetime('now')
% Define the output LAS file path
%%
% Read the LAS files
outputFile = 'merged_filedummy.las'; % Path for the merged LAS file

% Read the LAS files
tic
reader1 = lasFileReader(lasFile1);
reader2 = lasFileReader(lasFile2);
toc

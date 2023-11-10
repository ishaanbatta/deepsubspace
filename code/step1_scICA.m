% Enter the values for the variables required for the ICA analysis.
% Variables are on the left and the values are on the right.
% Characters must be enterd in single quotes

%% Modality
modalityType = 'smri';

%% Type of analysis
% Options are 1 and 2.
% 1 - Regular Group ICA
% 2 - Group ICA using icasso
which_analysis = 2;

%% ICASSO options.
% This variable will be used only when which_analysis variable is set to 2.
icasso_opts.sel_mode = 'randinit';  % Options are 'randinit', 'bootstrap' and 'both'
icasso_opts.num_ica_runs = 20; % Number of times ICA will be run
% icaOptions{1,1} = 'posact';
% icaOptions{1,2} = 'on';

%% Data file pattern. Include all subjects (3D images) in the file pattern or use char 
load('/data/qneuromark/Results/Subject_selection_str/ABCD/sub_info_ABCD_str.mat', 'T1subjlist_use')
file_name = 'Sm6mwc1pT1.nii';

Sub = length(T1subjlist_use); % number of subjects
input_data_file_patterns = cell(Sub,1);

for s_sub = 1:Sub
    temp_file = fullfile( T1subjlist_use{s_sub,1}, file_name); % functional image folder
    input_data_file_patterns(s_sub) = {temp_file};
end

%% Enter directory to put results of analysis
outputDir = '/data/qneuromark/Results/ICA_str/ABCD/low30';

%% Enter Name (Prefix) Of Output Files
prefix = 'SBM_ABCD';

%% Parallel info
% enter mode serial or parallel. If parallel, enter number of
% sessions/workers to do job in parallel
% parallel_info.mode = 'parallel';
% parallel_info.num_workers = 30;

%% Enter location (full file path) of the image file to use as mask
% or use Default mask which is []
maskFile = '/data/qneuromark/Results/Subject_selection_str/ABCD/comm_mask_ABCD_str.nii';

%% Batch Estimation. If 1 is specified then estimation of 
% the components takes place and the corresponding PC numbers are associated
% Options are 1 or 0
doEstimation = 0; 

%% Scale the components. Options are 0 and 2
% 0 - Don't scale
% 2 - Scale to Z scores
scaleType = 2;

%% 'Which ICA Algorithm Do You Want To Use';
% see icatb_icaAlgorithm for details or type icatb_icaAlgorithm at the
% command prompt.
% Note: Use only one subject and one session for Semi-blind ICA. Also specify atmost two reference function names

% 1 means infomax, 2 means fastICA, etc.
% algoType = 1;
algoType = 'MOO-ICAR';
refFiles = {'/data/qneuromark/Network_templates/Structural/Matched_template/Neuromark_v01_sMRI_low_30.nii'};
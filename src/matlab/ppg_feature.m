% --- PPG feature extraction ---

% Perform feature extraction on PPG signals and save the feature matrices.
% Read PPG signals from "....\data\02_processed_data\ppg", 
% perform feature extraction, and save the feature matrices to "....\data\03_features\ppg".

clc;
clear;
close all;

%% 1. Set Paths and Parameters
% --- Input/Output Paths ---
% Use relative paths to improve code portability
baseDataDir = fullfile('..', '..', 'data');
dataRoot = fullfile(baseDataDir, '02_processed_data', 'ppg');
outputRoot = fullfile(baseDataDir, '03_features', 'ppg');

% --- Parameter Settings ---
fs = 200; % Sampling rate (Hz)
secForFeature = 60; % Segment length for feature extraction (seconds)
winSize = round(fs * 5);  % Feature extraction window size (points)
winStride = round(fs * 5); % Feature extraction window stride (points)

%% 2. Preparation
% Create the output directory if it does not exist
if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
    fprintf('Created output directory: %s\n', outputRoot);
end

% Get all preprocessed PPG data files
csvFiles = dir(fullfile(dataRoot, 'ppg_*.csv'));
if isempty(csvFiles)
    error('No CSV files found in %s. Please check the path and file naming convention.', dataRoot);
end
fprintf('Found %d PPG files to process.\n\n', length(csvFiles));

%% 3. Loop Through Files and Extract Features
% Loop through each CSV file
for i = 1:length(csvFiles)
    csvName = csvFiles(i).name;
    fprintf('(%d/%d) Processing file: %s\n', i, length(csvFiles), csvName);
    
    try
        % --- Read Data ---
        data = readmatrix(fullfile(dataRoot, csvName));
        
        % Check if the data is long enough
        if size(data, 1) < fs * secForFeature
            warning('File %s does not have enough data (needs %d samples, has %d). Skipping.', ...
                csvName, fs * secForFeature, size(data, 1));
            continue;
        end
        
        % --- Signal Selection ---
        % Select the last 60-second segment for feature extraction
        ppgSignal = data(end - fs * secForFeature + 1 : end, 1);
        
        % ======================= Feature Extraction =======================
        
        %% 1. Waveform Features
        [means, sd, meanAbsDiff, meanAbsDiffNor, meanAbsDiff2, meanAbsDiff2Nor] ...
            = utils.statisticalFeature(ppgSignal, winSize, winStride);
        
        [nonStationarityIndex, fractalDimension] ...
            = utils.physicalFeature(ppgSignal, winSize, winStride);
        
        [heartBeatPSDEnergy] ...
            = utils.PSDEnergy_ppg(ppgSignal, winSize, winStride, fs);
        
        [meanFrequencies, meanSquaredAmplitudes] ...
            = utils.HHTFreAmp(ppgSignal, winSize, winStride, fs, 4);
            
        %% 2. HRV Features
        [ppgHRVtime, ppgHRVnonlinear] = ...
            utils.HRVfeature_ppg(ppgSignal, winSize, winStride, fs);
        
        % ======================= Feature Integration =======================
        
        % Statistical features
        statistical_features = {means, sd, meanAbsDiff, meanAbsDiffNor, meanAbsDiff2, meanAbsDiff2Nor};
        statFea = cell2mat(cellfun(@transpose, statistical_features, 'UniformOutput', false));
        
        % Physical features
        physical_features = {nonStationarityIndex, fractalDimension};
        phyTimeFea = cell2mat(cellfun(@transpose, physical_features, 'UniformOutput', false));
        
        % PSD features
        psd_features = {heartBeatPSDEnergy};  
        PSDFea = cell2mat(cellfun(@transpose, psd_features, 'UniformOutput', false));
        
        % Hilbert-Huang Transform (HHT) features
        hht_features = {meanFrequencies, meanSquaredAmplitudes}; 
        HHTFea = cell2mat(cellfun(@transpose, hht_features, 'UniformOutput', false));
        
        % HRV time-domain features
        timeHRV_features = {
                   ppgHRVtime.meanNN, ppgHRVtime.medianNN, ...
                   ppgHRVtime.SDNN, ppgHRVtime.RMSSD, ...
                   ppgHRVtime.PNN50, ppgHRVtime.meanRate, ...
                   ppgHRVtime.sdRate, ppgHRVtime.HRVTi
                   };
        timeHRVFea = cell2mat(cellfun(@transpose, timeHRV_features, 'UniformOutput', false));
        
        % HRV non-linear features
        nonlinearHRV_features = {
                       ppgHRVnonlinear.PoincareSD1
                       };
        nonlinearHRVFea = cell2mat(cellfun(@transpose, nonlinearHRV_features, 'UniformOutput', false));
        
        % --- Combine all features into a single matrix ---
        featureMatrix = [
                        statFea, ...
                        phyTimeFea, ...
                        PSDFea, ...
                        HHTFea, ...
                        timeHRVFea, ...
                        nonlinearHRVFea
                      ];
        
        %% --- Save Results ---
        % Generate output filename from input filename, e.g., "ppg_P05_09.csv" -> "ppgFea_P05_09.mat"
        [~, baseName, ~] = fileparts(csvName);
        outputBaseName = strrep(baseName, 'ppg_', 'ppgFea_');
        outputFileName = [outputBaseName, '.mat'];
        outputPath = fullfile(outputRoot, outputFileName);
        
        save(outputPath, 'featureMatrix');
        fprintf('    -> Saved PPG features to: %s\n', outputPath);
        
    catch ME
        % If an error occurs, print a warning and continue to the next file
        warning('An error occurred while processing file %s: %s', csvName, ME.message);
        continue;
    end
end

disp(' ');
disp('=====================================');
disp('  PPG feature extraction complete');
disp('=====================================');


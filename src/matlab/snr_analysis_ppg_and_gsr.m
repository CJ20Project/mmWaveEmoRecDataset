clear;
clc;
close all;

%% --- 1. Set Parameters ---
% File Paths
baseDir = '..\..\data\01_raw_data\ppg_and_gsr'; 

% Signal Parameters
fs = 200; % Sampling frequency (Hz)
ppg_cutoff_freq = 15; % Cutoff frequency for PPG signal and noise (Hz)
gsr_cutoff_freq = 1;  % Cutoff frequency for GSR signal and noise (Hz)

%% --- 2. Find and Initialize ---
% Find all folders starting with 'P'
participantDirs = dir(fullfile(baseDir, 'P*'));
participantDirs = participantDirs([participantDirs.isdir]);

if isempty(participantDirs)
    error('No participant folders found in the specified directory "%s". Please check the path.', baseDir);
end

fprintf('Found %d participants.\nStarting data processing...\n', length(participantDirs));

% Initialize struct arrays to store results, now including medianSnr
ppgResults = struct('participant', {}, 'meanSnr', {}, 'stdSnr', {}, 'medianSnr', {}, 'trialCount', {});
gsrResults = struct('participant', {}, 'meanSnr', {}, 'stdSnr', {}, 'medianSnr', {}, 'trialCount', {});

% Initialize arrays to collect ALL trial SNRs across all participants
allPpgTrialSnrs = [];
allGsrTrialSnrs = [];

%% --- 3. Loop Through Each Participant for Data Processing ---
for i = 1:length(participantDirs)
    participantName = participantDirs(i).name;
    participantPath = fullfile(baseDir, participantName);
    
    fprintf('Processing data of participant: %s...\n', participantName);
    
    % Find all CSV files in the current participant's folder
    csvFiles = dir(fullfile(participantPath, '*.csv'));
    
    % Initialize arrays to store SNR values for all trials of this participant
    ppgSnrValues = [];
    gsrSnrValues = [];
    
    % Loop through each CSV file for this participant
    for j = 1:length(csvFiles)
        filePath = fullfile(participantPath, csvFiles(j).name);
        try
            data = readmatrix(filePath, 'HeaderLines', 2);
            if size(data, 2) >= 2 && size(data, 1) > 10
                ppg_data = data(:, 1);
                gsr_data = data(:, 2);
                ppg_data = ppg_data(~isnan(ppg_data));
                gsr_data = gsr_data(~isnan(gsr_data));
                
                if ~isempty(ppg_data)
                    ppgSnrValues = [ppgSnrValues, calculateSnr(ppg_data, fs, ppg_cutoff_freq)];
                end
                if ~isempty(gsr_data)
                    gsrSnrValues = [gsrSnrValues, calculateSnr(gsr_data, fs, gsr_cutoff_freq)];
                end
            else
                warning('File %s has incorrect format or insufficient data. Skipping.', filePath);
            end
        catch ME
            warning('Could not process file %s. Error: %s', filePath, ME.message);
        end
    end
    
    % Accumulate all trial SNRs for the grand percentile calculation
    allPpgTrialSnrs = [allPpgTrialSnrs, ppgSnrValues];
    allGsrTrialSnrs = [allGsrTrialSnrs, gsrSnrValues];
    
    % Store the statistical results for this participant
    if ~isempty(ppgSnrValues)
        idx = length(ppgResults) + 1;
        ppgResults(idx).participant = participantName;
        ppgResults(idx).meanSnr     = mean(ppgSnrValues);
        ppgResults(idx).stdSnr      = std(ppgSnrValues);
        ppgResults(idx).medianSnr   = median(ppgSnrValues); % Calculate and store median
        ppgResults(idx).trialCount  = length(ppgSnrValues);
    end
    
    if ~isempty(gsrSnrValues)
        idx = length(gsrResults) + 1;
        gsrResults(idx).participant = participantName;
        gsrResults(idx).meanSnr     = mean(gsrSnrValues);
        gsrResults(idx).stdSnr      = std(gsrSnrValues);
        gsrResults(idx).medianSnr   = median(gsrSnrValues); % Calculate and store median
        gsrResults(idx).trialCount  = length(gsrSnrValues);
    end
end

fprintf('\nAll data processing complete. Generating report...\n\n');

%% --- 4. Report Results ---
% --- PPG Results Report ---
fprintf('=======================================================================\n');
fprintf('                     PPG Signal SNR Analysis Results\n');
fprintf('=======================================================================\n');
if ~isempty(ppgResults)
    % Per-participant results
    fprintf('--- Per-Participant Statistics ---\n');
    for k = 1:length(ppgResults)
        fprintf('%-5s: Mean = %6.2f dB, Median = %6.2f dB, SD = %5.2f dB (%d clips)\n', ...
            ppgResults(k).participant, ppgResults(k).meanSnr, ppgResults(k).medianSnr, ppgResults(k).stdSnr, ppgResults(k).trialCount);
    end
    
    % Overall summary
    allPpgMeanSnrs = [ppgResults.meanSnr];
    allPpgStdSnrs  = [ppgResults.stdSnr];
    allPpgMedianSnrs = [ppgResults.medianSnr];
    
    fprintf('\n--- Overall Summary for PPG across all participants ---\n');
    fprintf('  - Grand Average of Mean SNRs: %.2f dB\n', mean(allPpgMeanSnrs));
    fprintf('  - Range of Mean SNRs        : %.2f to %.2f dB\n', min(allPpgMeanSnrs), max(allPpgMeanSnrs));
    fprintf('  - Range of Median SNRs      : %.2f to %.2f dB\n', min(allPpgMedianSnrs), max(allPpgMedianSnrs));
    fprintf('  - Range of Std Devs         : %.2f to %.2f dB\n', min(allPpgStdSnrs), max(allPpgStdSnrs));
    fprintf('  - 5th Percentile: The lowest 5%% of SNRs are below %.2f dB.\n', prctile(allPpgTrialSnrs, 5));
    fprintf('=======================================================================\n\n');
else
    fprintf('No valid PPG data found.\n\n');
end

% --- GSR Results Report ---
fprintf('=======================================================================\n');
fprintf('                     GSR Signal SNR Analysis Results\n');
fprintf('=======================================================================\n');
if ~isempty(gsrResults)
    % Per-participant results
    fprintf('--- Per-Participant Statistics ---\n');
    for k = 1:length(gsrResults)
        fprintf('%-5s: Mean = %6.2f dB, Median = %6.2f dB, SD = %5.2f dB (%d clips)\n', ...
            gsrResults(k).participant, gsrResults(k).meanSnr, gsrResults(k).medianSnr, gsrResults(k).stdSnr, gsrResults(k).trialCount);
    end
    
    % Overall summary
    allGsrMeanSnrs = [gsrResults.meanSnr];
    allGsrStdSnrs  = [gsrResults.stdSnr];
    allGsrMedianSnrs = [gsrResults.medianSnr];

    fprintf('\n--- Overall Summary for GSR across all participants ---\n');
    fprintf('  - Grand Average of Mean SNRs: %.2f dB\n', mean(allGsrMeanSnrs));
    fprintf('  - Range of Mean SNRs        : %.2f to %.2f dB\n', min(allGsrMeanSnrs), max(allGsrMeanSnrs));
    fprintf('  - Range of Median SNRs      : %.2f to %.2f dB\n', min(allGsrMedianSnrs), max(allGsrMedianSnrs));
    fprintf('  - Range of Std Devs         : %.2f to %.2f dB\n', min(allGsrStdSnrs), max(allGsrStdSnrs));
    fprintf('  - 5th Percentile: The lowest 5%% of SNRs are below %.2f dB.\n', prctile(allGsrTrialSnrs, 5));
    fprintf('=======================================================================\n');
else
    fprintf('No valid GSR data found.\n');
end

%% --- SNR Calculation Function ---
function snr_db = calculateSnr(data, fs, cutoff_freq)
    % This function estimates SNR by separating the signal into low-frequency (signal)
    % and high-frequency (noise) components using a Butterworth filter.
    n = 3; % Filter order (3rd order)
    nyquist_freq = fs / 2;
    Wn = cutoff_freq / nyquist_freq;
    
    [b_low, a_low] = butter(n, Wn, 'low');
    [b_high, a_high] = butter(n, Wn, 'high');
    
    signal_component = filtfilt(b_low, a_low, data);
    noise_component = filtfilt(b_high, a_high, data);
    
    ms_signal = mean(signal_component.^2);
    ms_noise = mean(noise_component.^2);
    
    if ms_noise < eps % Avoid division by zero
        snr_db = Inf;
    else
        snr_db = 10 * log10(ms_signal / ms_noise);
    end
end
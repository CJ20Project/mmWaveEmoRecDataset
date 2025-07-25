function [nonStationarityIndex, fractalDimension] ...
    = physicalFeature(heartBeat, winSize, winStride)

% Initialize output arrays
numWindows = floor((length(heartBeat) - winSize) / winStride) + 1;
nonStationarityIndex = zeros(1, numWindows);
fractalDimension = zeros(1, numWindows);
    
for i = 1:numWindows
    % Get the data for the current window
    startIdx = (i-1) * winStride + 1;
    endIdx = startIdx + winSize - 1;
    currentWindow = heartBeat(startIdx : endIdx);
    
    % Normalize the data: mean = 0, std = 1
    normalizedWindow = zscore(currentWindow);
    
    % Divide the data into 20 segments of roughly equal length
    segmentLength = ceil(length(normalizedWindow) / 20);
    means = zeros(20, 1);
    
    for j = 1 : 20
        % Determine the start and end indices for the current segment
        segStart = (j - 1) * segmentLength + 1;
        segEnd = min(j * segmentLength, length(normalizedWindow));
        
        % Calculate the mean of the current segment
        if segStart <= segEnd
            means(j) = mean(normalizedWindow(segStart : segEnd));
        else
            % If the segment is empty, set to NaN
            means(j) = NaN; 
        end
    end
    
    % Calculate the standard deviation of these 20 means
    validMeans = means(~isnan(means)); % Ignore NaN values
    nonStationarityIndex(i) = std(validMeans);
 
    % Calculate the Higuchi Fractal Dimension
    [fractalDimension(i), ~] = HFD_LCALC(normalizedWindow);
end
end

%% 
function [ Df,x,y,kmax,LARE ] = HFD_LCALC( data, varargin)
%HFD_LCALC calculates the Curve Lengths for various series of curves
%obtained from the input data
%   Based on Higuchi Fractal Dimension Algorithm, Df (fractal dimension)
%   can be calculated from L(k)~k^(-Df).
%   This program calculated L(k) for various k=1 to kmax
%INPUT ARGUMENTS:
%   data: Mandatory argument and should be row vector. If not, it will be
%         converted to row vector
%   kmax: If not specfied takes floor(length(data)/2) by default. If
%         specified it should be less or equal to floor(length(data)/2).
%         Else it takes the default value. (It should be an integer).
% FAST version without any loss of accuracy in HFD calculation
% by: J. Wanliss, Ph.D 6/2021

%Error check and Intialize the variables, arrays
nVar=length(varargin);
if(nVar>1)
    error('Too many Inputs::Expects only input data, optional kmax, and plot status');
end
[r, c]=size(data);
if (r>1)
    % fprintf('Converting input data to a row vector\n');
    data=data(:)';% data needs to be a vector
end
L=length(data);
kmax=[];
if (nVar==1)
    kmax=varargin{1};
end
[r, c]=size(kmax);
if((r~=1) || (c~=1) || (isnumeric(kmax)==0) || (kmax>floor(L/2)) || (kmax<1))
    kmax=floor(L/2);
    % fprintf('kmax not a proper value. Changing kmax= %d\n',kmax);
end
oldkmax=kmax;
lastval = log10(round(length(data)/2));
kmax=unique(round(logspace(0.3,lastval,40)));
kmax=[1 kmax(kmax<=oldkmax)];
lkmax=length(kmax);

LARE=zeros(1,length(kmax));%LARE stores the length of curves L(k) for k=1:kmax
y=LARE;%this is to store y=log(LARE)
x=LARE;%this is to store x=log(1/k)

count = 1;
for k=kmax
    LAk=0;
    for i=1:k
        LAi=0;
        for j=1:floor((L-i)/k)        
            LAi=LAi+abs(data(i+j*k)-data(i+(j-1)*k));
        end
        a=(L-1)/(floor((L-i)/k)*k);
        LAk=LAk+LAi*a/k;
    end
    LARE(count)=LAk/k;
    y(count)=log(LARE(count));
    x(count)=log(1/k);
    count=count+1;
end

%Df=(max(kmax)*sum(x.*y)-sum(x)*sum(y))/(max(kmax)*sum(x.*x)-sum(x)*sum(x));
%Df = NaN;
% % Create figure
% figure1 = figure;
% % Create axes
% axes1 = axes('Parent',figure1);
% box(axes1,'on');
% hold(axes1,'all');
% plot(x,y)
% title('Plot Log(1/k) vs Log(L(k))')
% xlabel('Log(1/k)')
% ylabel('Log(L(k))')
% str=['Df=',num2str(Df)];
% annotation(figure1,'textbox',...
%     [0.8375 0.9262 0.1518 0.0667],...
%     'String',{str},...
%     'FitBoxToText','on');


%using polyfit 
coef=polyfit(x(2:length(x)),y(2:length(x)),1);
Df=coef(1);
end

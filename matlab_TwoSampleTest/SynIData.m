function [ synIdata ] = SynIData( ShiftMagnitude, BatchSize, seed)
%SynIDATA generate Gaussian data with shifting mean
%   Detailed explanation goes here
%   ShiftMagnitude
%   BatchSize

if seed > 0
    rng(seed)
end

sigma = 1;
synIdata = normrnd(0, sigma, [BatchSize,2]);

for m = ShiftMagnitude
    
    group1 = normrnd(0 + m, sigma, [BatchSize,2]);
    
    data = [group1];
    data = data(randperm(length(data)), :);
    
    synIdata = [synIdata;data];
    
end
rng('shuffle');
end


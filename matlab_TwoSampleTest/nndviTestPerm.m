function [ testStat,thresh,params ] = nndviTestBoot( X,Y,alpha,params )
%IPDTESTBOOT Summary of this function goes here
%   Detailed explanation goes here

mx = size(X,1);
my = size(Y,1);
X_ind = 1:mx;
Y_ind = mx+1:mx+my;

if params.knn==0
    knn = selectK(X, Y);
    params.knn = knn;
else
    knn = params.knn;
end

dataTotal = [X; Y];
expandMatrix = 0:1:(mx+my-1);
expandMatrix = expandMatrix.*(mx+my);
expandMatrix = expandMatrix';
expandMatrix = repmat(expandMatrix, [1, knn]);
knnMatrix = knnsearch(dataTotal, dataTotal, 'K', knn);
knnIndex = knnMatrix+ expandMatrix;
adjMatrix = zeros(mx+my,mx+my);
adjMatrix(knnIndex) = 1;

normKnnMatrix = sum(adjMatrix, 2);
normKnnMatrix = repmat(normKnnMatrix, [1 size(adjMatrix, 2)]);
normKnnMatrix = adjMatrix./normKnnMatrix;

v1 = normKnnMatrix(X_ind, :);
v1 = sum(v1);

v = sum(normKnnMatrix);
v2 = v - v1;

% dissimilarity
testStat = sum(abs(v1-v2)./v/(mx+my));

% permutation
NNParr = zeros(params.shuff,1);
rng((mx+my)*2+params.shuff);
for whichSh=1:params.shuff
    
    [notUsed,shuffInd] = sort(rand(mx+my,1));
    
    X_ind_shuff = shuffInd(1:mx);
    Y_ind_shuff = shuffInd(mx+1:mx+my);
    
    vX = sum(normKnnMatrix(X_ind_shuff, :),1);
    vY = sum(normKnnMatrix(Y_ind_shuff, :),1);
    
    NNParr(whichSh) = sum(abs(vX-vY)./v/(mx+my));
    
end
rng('shuffle');

NNParr = sort(NNParr);
thresh = NNParr(round((1-alpha)*params.shuff));

end


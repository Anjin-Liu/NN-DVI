function [ knn ] = selectK( X, Y )
%SELECTBETA Summary of this function goes here
%   Detailed explanation goes here

mx = size(X,1);
my = size(Y,1);
m = mx+my;

kValues = [1:5:mx];
counter=1;
indicatorVec = zeros(1,size(kValues,1));
minIndicator = 1;
dataTotal = [X; Y];
expandMatrix = 0:1:(mx+my-1);
expandMatrix = expandMatrix.*(mx+my);
expandMatrix = expandMatrix';

for k = kValues
    
    tempexpandMatrix = repmat(expandMatrix, [1, k]);
    knnMatrix = knnsearch(dataTotal, dataTotal, 'K', k);
    knnIndex = knnMatrix+ tempexpandMatrix;
    adjMatrix = zeros(mx+my,mx+my);
    adjMatrix(knnIndex) = 1;
    
    normKnnMatrix = sum(adjMatrix, 2);
    normKnnMatrix = repmat(normKnnMatrix, [1 size(adjMatrix, 2)]);
    tempdistMatrix = adjMatrix./normKnnMatrix;
   
    indep = zeros(1, mx+my);
    for j = 1:mx+my
        Pi = ones(1,mx+my)*(tempdistMatrix.*(tempdistMatrix(:,j)*ones(1,mx+my)));
        Pi = Pi./Pi(j);
        indep(j) = 1-mean(nonzeros(Pi));
        
    end
    indicatorVec(counter) = 1-mean(indep);
    if minIndicator > indicatorVec(counter)
        minIndicator = indicatorVec(counter);
        knn = k;
    end
    counter = counter +1;
end

sum(tempdistMatrix(1,:));

figure
plot([1:1:size(indicatorVec,2)], indicatorVec);

end


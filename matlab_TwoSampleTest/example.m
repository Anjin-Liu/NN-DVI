alpha =0.05
testTimeMax = 100
delta = [0, 0.2, 0.4];
batchSizeArray = [50, 100, 500];
data = zeros(4, length(delta)*length(batchSizeArray));

deltaRun = [1, 2, 3];
batchSizeRun = [2];

for deltaIndex = deltaRun
    for batchSizeIndex = batchSizeRun
        
        batchSize = batchSizeArray(batchSizeIndex);
        hArr_nndvi = zeros(testTimeMax,1);
        p_nndvi = nndviParams;

        fprintf('Dataset detla: %.2f, batchSize: %.2d\n', delta(deltaIndex),  batchSize);
        
        for testTime = 1:1:testTimeMax
            
            randSeed = testTime;
            SynData = SynIData(delta,batchSize,randSeed);
            
            data_sta = deltaIndex*batchSize+1;
            data_end = deltaIndex*batchSize+batchSize;
            
            [s_nndvi, t_nndvi, p_nndvi] = nndviTestPerm(SynData(1:batchSize,:), SynData(data_sta:data_end,:), alpha, p_nndvi);
            h_nndvi= s_nndvi>t_nndvi;
            hArr_nndvi(testTime) = h_nndvi;

            fprintf('  testTime: %d, h_nndvi sum: %d\n', testTime, sum(hArr_nndvi));
            
        end  
    end
end


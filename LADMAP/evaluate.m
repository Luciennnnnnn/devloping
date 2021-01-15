function model = evaluate(dataset_name, ed, fraction, mu, sigma)
    [Y, outliers_p] = generator(dataset_name, fraction, mu, sigma);
    if ed == -1
        ed = size(Y,3);
    end
    Y = Y(:,:,1:ed);
    [M,m,n,p] = convert_video3d_to_2d(Y);
    model = run_algorithm('LRR','LADMAP',M);
    model.precision = 0;
    model.FPR = 0;
    
    model.TPRS = zeros(1,size(Y,3));
    model.FPRS = zeros(1,size(Y,3));
    model.RSE = zeros(1,size(Y,3));
    for i = 1:size(Y,3)
        [TPR, FPR] = check(model.S(:,i), outliers_p(:, :, i), sum(sum(outliers_p(:,:,i))), size(Y));
        model.precision = model.precision + TPR;
        model.FPR = model.FPR + FPR;
        model.TPRS(1,i) = TPR;
        model.FPRS(1,i) = FPR;
        temp1 = model.L(:,i) + model.S(:,i) - M(:,i);
        temp2 = M(:,i);
        model.RSE(1,i) = norm(temp1(:))/norm(temp2(:));
    end
    model.precision = model.precision / size(Y,3);
    model.FPR = model.FPR /size(Y,3);
end
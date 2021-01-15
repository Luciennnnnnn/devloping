function model = evaluate(dataset_name, ed, fraction, mu, sigma, R, forgetting_factor, sliding_window_size)
    [Y, outliers_p] = generator(dataset_name, fraction, mu, sigma);
    if ed == -1
        ed = size(Y,3);
    end
    model = brst_wrapper(Y(:,:,1:ed),1,forgetting_factor,sliding_window_size,R);
    model.precision = 0;
    model.FPR = 0;
    tmp = size(model.proposed_factorization);
    model.TPRS = zeros(1,tmp(2));
    model.FPRS = zeros(1,tmp(2));
    for i = 1:tmp(2)
        [TPR, FPR] = check(model.proposed_factorization{2,i}, outliers_p(:, :, i), sum(sum(outliers_p(:,:,i))), size(Y));
        model.precision = model.precision + TPR;
        model.FPR = model.FPR + FPR;
        model.TPRS(1,i) = TPR;
        model.FPRS(1,i) = FPR;
    end
    model.precision = model.precision / tmp(2);
    model.FPR = model.FPR / tmp(2);
end
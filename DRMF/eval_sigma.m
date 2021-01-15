function eval_sigma(dataset_name)
    dd = 10
    TPRS = zeros(1, dd);
    FPRS = zeros(1, dd);
    sigmas = 1:10;
    for i = 1:dd
        tic;
        model = evaluate(dataset_name, -1, 0.1, 0, sigmas(i));
        toc;
        TPRS(1,i) = model.precision;
        FPRS(1,i) = model.FPR;
    end

    % saving...
    save ['D:\results\',dataset_name,'\DRMF\sigma_TPRS.mat'] TPRS;
    save ['D:\results\',dataset_name,'\DRMF\sigma_FPRS.mat'] FPRS;
end
function eval_mu(dataset_name)
    dd = 10
    TPRS = zeros(1, dd);
    FPRS = zeros(1, dd);
    mus = 1:10;
    for i = 1:dd
        tic;
        model = evaluate(dataset_name, -1, 0.1, mus(i)/10, 0.1);
        toc;
        TPRS(1,i) = model.precision;
        FPRS(1,i) = model.FPR;
    end

    % saving...
    save(['D:\results\',dataset_name,'\noncvxRPCA\mu_TPRS.mat'],'TPRS');
    save(['D:\results\',dataset_name,'\noncvxRPCA\mu_FPRS.mat'],'TPRS');
end
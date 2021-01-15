function eval_ratio(dataset_name)
    dd = 10
    TPRS = zeros(1, dd);
    FPRS = zeros(1, dd);
    fractions = 1:10;
    for i = 1:dd
        tic;
        model = evaluate(dataset_name, -1, fractions(i)/100, 0, 0.1);
        toc;
        TPRS(1,i) = model.precision;
        FPRS(1,i) = model.FPR;
    end

    % saving...
    save(['D:\results\',dataset_name,'\noncvxRPCA\ratio_TPRS.mat'],'TPRS');
    save(['D:\results\',dataset_name,'\noncvxRPCA\ratio_FPRS.mat'],'TPRS');
end
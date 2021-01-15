function eval_ratio(dataset_name)
    dd = 10;
    TPRS = zeros(1, dd);
    FPRS = zeros(1, dd);
    ratios = 1:10;
    for i = 1:dd
        tic;
        model = evaluate(dataset_name, 8000, ratios(i)/100, 0, 0.1, 10, 0.98, 20);
        toc;
        TPRS(1,i) = model.precision;
        FPRS(1,i) = model.FPR;
    end

    % saving...
    save(['D:\results\',dataset_name,'\BRST\ratio_TPRS.mat'],'TPRS');
    save(['D:\results\',dataset_name,'\BRST\ratio_FPRS.mat'],'FPRS');
end
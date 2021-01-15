function eval_time(dataset_name)
    tic;
    model = evaluate(dataset_name, 8000, 0.1, 0, 0.1, 10, 0.98, 20);
    RSE = model.err_residual(2:end);
    TPRS = model.TPRS;
    FPRS = model.FPRS;
    toc
    % saving...
    save(['D:\results\',dataset_name,'\BRST\TPRS.mat'], 'TPRS');
    save(['D:\results\',dataset_name,'\BRST\FPRS.mat'], 'FPRS');
    save(['D:\results\',dataset_name,'\BRST\RSE.mat'], 'RSE');
end
function eval_mu(dataset_name)
    tic;
    model = evaluate(dataset_name, -1, 0.1, 0, 0.1);
    RSE = model.RSE;
    TPRS = model.TPRS;
    FPRS = model.FPRS;
    toc
    % saving...
    save(['D:\results\',dataset_name,'\noncvxRPCA\TPRS.mat'],'TPRS');
    save(['D:\results\',dataset_name,'\noncvxRPCA\FPRS.mat'],'FPRS');
    save(['D:\results\',dataset_name,'\noncvxRPCA\RSE.mat'],'RSE');
end
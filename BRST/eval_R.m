function eval_R(dataset_name)
    dd = 10
    TPRS = zeros(10, dd);
    FPRS = zeros(10, dd);
    Rs = 5:2:23;
    for i = 1:1
        for j = 1:1
            tic;
            model = evaluate(dataset_name, 48096, 0.1, 0, 0.1, Rs(j), 0.98, 20);
            toc;
            TPRS(i,j) = model.precision;
            FPRS(i,j) = model.FPR;
        end
    end
    T_means = zeros(1,dd);
    T_stderr = zeros(1,dd);
    F_means = zeros(1,dd);
    F_stderr = zeros(1,dd);
    for i = 1:dd
        T_means(i) = mean(TPRS(:, i));
        T_stderr(i) = std(TPRS(:, i)) / sqrt(10);
    end
    for i = 1:dd
        F_means(i) = mean(FPRS(:, i));
        F_stderr(i) = std(FPRS(:, i)) / sqrt(10);
    end
    % saving...
    save D:\results\Abilene\brst\R_TPRS_mean.mat T_means;
    save D:\results\Abilene\brst\R_TPRS_stderr.mat T_stderr;
    save D:\results\Abilene\brst\R_FPRS_mean.mat F_means;
    save D:\results\Abilene\brst\R_FPRS_stderr.mat F_stderr;
end
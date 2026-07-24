function run_damp_multidim(input_csv, output_csv, official_dir)
    arguments
        input_csv (1,1) string
        output_csv (1,1) string
        official_dir (1,1) string
    end

    set(groot, 'defaultFigureVisible', 'off');
    raw = readmatrix(input_csv);
    if size(raw, 2) < 2
        error('Expected one or more value columns followed by a label column.');
    end
    values = raw(:, 1:end-1);
    labels = raw(:, end);

    source_path = fullfile(official_dir, 'DAMP_Multidim.m');
    source = fileread(source_path);
    source = regexprep( ...
        source, ...
        'function\s+DAMP_Multidim\(T,dimension_num\)', ...
        'function Left_MP = DAMP_Multidim_export(T,dimension_num)', ...
        'once' ...
    );
    generated_dir = fullfile(fileparts(output_csv), 'generated_damp_adapter');
    if ~exist(generated_dir, 'dir')
        mkdir(generated_dir);
    end
    generated_path = fullfile(generated_dir, 'DAMP_Multidim_export.m');
    file_id = fopen(generated_path, 'w');
    cleanup = onCleanup(@() fclose(file_id));
    fprintf(file_id, '%s', source);
    clear cleanup;

    addpath(generated_dir);
    addpath(official_dir);
    scores = DAMP_Multidim_export(values, size(values, 2));
    writematrix([labels, scores], output_csv);
end

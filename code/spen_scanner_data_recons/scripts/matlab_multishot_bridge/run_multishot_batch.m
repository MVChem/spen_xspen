function run_multishot_batch(job_json, legacy_root, archive_root)
% Execute ordered frame jobs in one CPU MATLAB process; retain every log.
jobs = jsondecode(fileread(job_json));
for index = 1:numel(jobs)
    job = jobs(index);
    started = tic;
    result = struct('frame_id',job.frame_id,'status','failed');
    diary(job.log_path);
    try
        run_multishot_frame(job.input_mat,job.output_mat,legacy_root,archive_root);
        result.status = 'completed';
    catch problem
        fprintf('%s\n',getReport(problem,'extended','hyperlinks','off'));
        result.error = problem.message;
    end
    diary off;
    output_text = fileread(job.log_path);
    result.optimization_budget_reached = contains(output_text,'Maximum number of function evaluations') ...
        || contains(output_text,'Maximum number of iterations');
    result.phase_fit_fallback = contains(output_text,'Phase fitting failed');
    result.badly_conditioned_polynomial = contains(output_text,'Polynomial is badly conditioned');
    result.elapsed_seconds = toc(started);
    result.matlab_version = version;
    file = fopen([job.result_json '.tmp'],'w');
    fprintf(file,'%s\n',jsonencode(result));
    fclose(file);
    movefile([job.result_json '.tmp'],job.result_json,'f');
    fprintf('%s\n',jsonencode(result));
    close all force;
end
end

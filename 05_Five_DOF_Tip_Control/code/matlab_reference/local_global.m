function [p_global, h_global, p_e_global, quat_e_global] = local_global(p_total, h_total, p_e, quat_e, params)
    %#codegen
    % Initialize output arrays
    p_global = zeros(size(p_total));
    h_global = zeros(size(h_total));
    p_e_global = zeros(size(p_e));
    quat_e_global = zeros(size(quat_e));

    % Transformation for catheter states
    for step = 1:length(p_e_global)
        for point = 1:params.num_points
            % Position: p_global = R_t * p_local + T_t
            p_local = p_total(point, :, step)';
            p_global(point, :, step) = (params.R_t * p_local + params.T_t)';

            % Orientation transformation (quaternion to rotation matrix approach)
            h_local = h_total(point, :, step);
            R_local = quat2rotm(h_local);         % Local orientation matrix
            R_global = params.R_t * R_local;             % Apply global rotation
            h_global(point, :, step) = rotm2quat(R_global)';  % Convert back to quaternion
        end

        % External magnet position: p_e_global = R_t * p_e + T_t
        p_e_local = p_e(step, :)';
        p_e_global(step, :) = (params.R_t * p_e_local + params.T_t)';

        % External magnet orientation: eul_e_global
        R_e_local = quat2rotm(quat_e(step, :));
        R_e_global = params.R_t * R_e_local;
        quat_e_global(step, :) = rotm2quat(R_e_global);
    end
end

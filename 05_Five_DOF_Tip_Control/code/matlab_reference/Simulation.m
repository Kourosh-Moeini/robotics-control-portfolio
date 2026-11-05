classdef Simulation < handle
    properties
        % Simulation configurations
        num_points
        s_length
        ds
        steps
        dt
        params
        
        % State tracking
        sol_prev
        sol_prev_rand
        step_counter
        
        % Pre-allocated history arrays
        p_total
        h_total
        p_e_history
        quat_e_history
    end
    
    methods
        function obj = Simulation(matter, mag_set)
            % Constructor: Initialize simulation parameters
            switch matter
                case 'Rod1'
                    obj.num_points = 20;
                    obj.s_length = 0.03;
                case 'Rod2'
                    obj.num_points = 50;
                    obj.s_length = 0.1;
                case 'PDMS1'
                    obj.num_points = 50;
                    obj.s_length = 0.10669;
                case 'PDMS2'
                    obj.num_points = 50;
                    obj.s_length = 0.105;
                otherwise
                    error('Unknown material specified.');
            end
            
            obj.steps = 500;
            obj.dt = 0.01;
            
            s = linspace(0, obj.s_length, obj.num_points);
            obj.ds = s(2) - s(1);

            % Set material and magnet parameters
            obj.params = parameters(matter);
            obj.params.num_points = obj.num_points;
            obj.params.s_length = obj.s_length;
            obj.params.steps = obj.steps;
            obj.params.ds = obj.ds;
            obj.params.s = s;   
            
            obj.params = mag_parameters(mag_set, obj.params);

            % Set transformation parameters
            obj.params.eul_t = [0.0, deg2rad(90), deg2rad(-90)];
            obj.params.H_t = eul2quat(obj.params.eul_t, "ZYX");
            obj.params.R_t = quat2rotm(obj.params.H_t);
            obj.params.T_t = [0.0; 0.0; 0.0];

            obj.params.invKse = inv(obj.params.Kse);
            obj.params.invKbt = inv(obj.params.Kbt);
            obj.params.f_global = obj.params.rho * obj.params.A * obj.params.g; 

            % Define the simulation setup for Python reference
            obj.params.matter = matter;
            obj.params.mag_set = mag_set; 
        end

        function [pos0, dir0, reset_terminated] = reset(obj, action0)
            % Extract action directly
            p_e0 = action0(1:3);
            quat_e0 = action0(4:7);

            % Pre-allocate state arrays efficiently based on obj.steps
            obj.p_total = zeros(obj.num_points, 3, obj.steps + 1);
            obj.p_total(:, 3, 1) = obj.params.s(:); % Initial pos along z

            obj.h_total = zeros(obj.num_points, 4, obj.steps + 1);
            obj.h_total(:, :, 1) = repmat([1, 0, 0, 0], obj.num_points, 1);

            obj.step_counter = 1;

            % Pre-allocate histories
            obj.p_e_history = zeros(obj.steps + 1, 3);
            obj.quat_e_history = zeros(obj.steps + 1, 4);
            
            obj.p_e_history(1, :) = p_e0;
            obj.quat_e_history(1, :) = quat_e0;

            % Solve BVP
            [sol, p, h, ~, terminated] = bvp_solver(p_e0, quat_e0, obj.params, []);

            % Store valid base solutions
            obj.sol_prev = sol;
            obj.sol_prev_rand = sol; % Base for rand_setpoint

            % Store initial tip state
            obj.p_total(:, :, 1) = p;
            obj.h_total(:, :, 1) = h;

            % Outputs for Python (Ensure dir is a 1x3 row vector)
            pos0 = p(end, :);
            R_tip0 = my_quat2rotm(h(end, :));
            dir0 = R_tip0(:, 3)'; 
            
            % Logic check for Reset Terminated
            terminate_action0 = norm(p(end, :) - p_e0) > 0.15;
            reset_terminated = double(terminate_action0 || terminated);
        end

        function [next_pos, next_dir, next_b, terminated] = step(obj, action)
            % Advance simulation smoothly without recording history (training mode)
            [obj.sol_prev, p, h, b, term] = bvp_solver(action(1:3), action(4:7), obj.params, obj.sol_prev);

            % Prepare outputs
            next_pos = p(end, :);
            next_R = my_quat2rotm(h(end, :));
            next_dir = next_R(:, 3)';
            next_b = b;
            terminated = double(term);
        end

        function [next_pos, next_dir, target_term] = rand_setpoint(obj, action)
            % Generate random setpoint target using its own independent solver state
            [sol, p, h, ~, term] = bvp_solver(action(1:3), action(4:7), obj.params, obj.sol_prev_rand);

            % Prepare outputs
            next_pos = p(end, :);
            next_R = my_quat2rotm(h(end, :));
            next_dir = next_R(:, 3)';
            
            % Check magnetic threshold
            target_term = double(term);
            
            % Only update the rand baseline solution if it wasn't a failure
            if ~target_term
                obj.sol_prev_rand = sol; 
            end
        end

        function [next_pos, next_dir, next_b, terminated] = test_model(obj, action)
            % Step & track histories for Video/Snapshot Rendering
            p_e = action(1:3);
            quat_e = action(4:7);

            obj.step_counter = obj.step_counter + 1;
            
            % Safeguard: dynamically expand arrays if python takes more steps than pre-allocated
            if obj.step_counter > size(obj.p_total, 3)
                expansion_size = 100;
                obj.p_total = cat(3, obj.p_total, zeros(obj.num_points, 3, expansion_size));
                obj.h_total = cat(3, obj.h_total, zeros(obj.num_points, 4, expansion_size));
                obj.p_e_history = [obj.p_e_history; zeros(expansion_size, 3)];
                obj.quat_e_history = [obj.quat_e_history; zeros(expansion_size, 4)];
            end

            % Record external magnet history
            obj.p_e_history(obj.step_counter, :) = p_e;
            obj.quat_e_history(obj.step_counter, :) = quat_e;

            % Solve BVP
            [obj.sol_prev, p, h, b, term] = bvp_solver(p_e, quat_e, obj.params, obj.sol_prev);

            % Record rod kinematics history
            obj.p_total(:, :, obj.step_counter) = p;
            obj.h_total(:, :, obj.step_counter) = h;

            % Prepare outputs
            next_pos = p(end, :);
            next_R = my_quat2rotm(h(end, :));
            next_dir = next_R(:, 3)';      

            next_b = b;
            terminated = double(term);
        end

        function create_video(obj, video_dir)
            % Retrieve only the valid recorded steps (avoids trailing zeros)
            [v_p_total, v_h_total, v_p_e_history, v_quat_e_history] = obj.get_valid_history();

            % Convert local coordinates to global for plotting
            [p_global, h_global, p_e_global, quat_e_global] = local_global(...
                v_p_total, v_h_total, v_p_e_history, v_quat_e_history, obj.params);

            % Prepare parameters for plotting
            plot_params = obj.params;
            plot_params.p_e = p_e_global;
            plot_params.quat_e = quat_e_global;
            plot_params.dt = obj.dt;

            % Call the plotting function
            plot_trajectory_as_vid(p_global, h_global, p_e_global, quat_e_global, plot_params, video_dir);
        end

        % function create_snapshot(obj, folder_dir, target_pos)
        %     % Retrieve only the valid recorded steps (avoids trailing zeros)
        %     [v_p_total, v_h_total, v_p_e_history, v_quat_e_history] = obj.get_valid_history();
        % 
        %     % Convert local coordinates to global for plotting
        %     [p_global, h_global, p_e_global, quat_e_global] = local_global(...
        %         v_p_total, v_h_total, v_p_e_history, v_quat_e_history, obj.params);
        % 
        %     target_pos = (obj.params.R_t * target_pos(:) + obj.params.T_t)';
        % 
        %     % Prepare parameters for plotting
        %     plot_params = obj.params;
        %     plot_params.p_e = p_e_global;
        %     plot_params.quat_e = quat_e_global;
        %     plot_params.dt = obj.dt;
        % 
        %     % old_filename = folder_dir;
        %     % 
        %     % % Extract the path, name, and extension
        %     % [filepath, name, ext] = fileparts(old_filename);
        %     % 
        %     % % Create the new filename with a .mat extension
        %     % new_filename = fullfile(filepath, [name, '.mat']);
        %     % 
        %     % % Saves all workspace variables to a file named 'my_workspace.mat'
        %     % save(new_filename, 'p_global', 'h_global', ...
        %     %     'p_e_global', 'quat_e_global', 'target_pos', 'plot_params', 'folder_dir')
        % 
        %     % Call the plotting function with global coordinates
        %     plot_reaching_snapshots(p_global, h_global, p_e_global, quat_e_global, target_pos, plot_params, folder_dir)
        % end

        % function create_snapshot(obj, folder_dir, targets)
        %     % Retrieve only the valid recorded steps (avoids trailing zeros)
        %     [v_p_total, v_h_total, v_p_e_history, v_quat_e_history] = obj.get_valid_history();
        % 
        %     % Convert local coordinates to global for plotting
        %     [p_global, h_global, p_e_global, quat_e_global] = local_global(...
        %         v_p_total, v_h_total, v_p_e_history, v_quat_e_history, obj.params);
        % 
        %     % Apply rotation and translation to target positions and directions
        %     % Note: targets(:, 1:3)' makes it 3xN for R_t multiplication, then ' makes it Nx3 again
        %     target_pos = (obj.params.R_t * targets(:, 1:3)' + obj.params.T_t)';
        %     target_dir = (obj.params.R_t * targets(:, 4:6)')';
        % 
        %     % Recombine into an Nx6 matrix for the plotting function
        %     targets = [target_pos, target_dir];
        % 
        %     % Prepare parameters for plotting
        %     plot_params = obj.params;
        %     plot_params.p_e = p_e_global;
        %     plot_params.quat_e = quat_e_global;
        %     plot_params.dt = obj.dt;
        % 
        %     old_filename = folder_dir;
        % 
        %     % Extract the path, name, and extension
        %     [filepath, name, ext] = fileparts(old_filename);
        % 
        %     % Create the new filename with a .mat extension
        %     new_filename = fullfile(filepath, [name, '.mat']);
        % 
        %     % Saves all workspace variables to a file named 'my_workspace.mat'
        %     save(new_filename, 'p_global', 'h_global', ...
        %         'p_e_global', 'quat_e_global', 'targets', 'plot_params', 'folder_dir')
        % 
        %     % Call the 3D snapshot plotting function
        %     plot_tracking_snapshots(p_global, h_global, p_e_global, quat_e_global, targets, plot_params, folder_dir);
        % end

    end
    
    methods (Access = private)
        function [v_p, v_h, v_pe, v_qe] = get_valid_history(obj)
            % Slices the pre-allocated matrices to only return populated steps
            idx = 1:obj.step_counter;
            v_p = obj.p_total(:, :, idx);
            v_h = obj.h_total(:, :, idx);
            v_pe = obj.p_e_history(idx, :);
            v_qe = obj.quat_e_history(idx, :);
        end
    end
end

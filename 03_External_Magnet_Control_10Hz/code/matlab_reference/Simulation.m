classdef Simulation < handle
    properties
        ds
        dt
        z
        num_points
        s_length
        z_rand
        steps
        params
        p_total
        h_total
        q_total
        w_total
        sol_prev
        sol_prev_rand
        step_counter
        p_e_history % External magnet positions history (local)
        quat_e_history % External magnet orientations history (local)
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
                    obj.s_length = 0.120;
                    obj.s_length = 0.105;
            end

            obj.steps = 500;
            obj.dt = 0.01;
            % obj.dt = 0.1;
            s = linspace(0, obj.s_length, obj.num_points);
            obj.ds = s(2) - s(1);

            % Get material parameters
            obj.params = parameters(matter);
            obj.params.num_points = obj.num_points;
            obj.params.s_length = obj.s_length;
            obj.params.steps = obj.steps;
            obj.dt = obj.dt;
            obj.params.ds = obj.ds;
            obj.params.s = s;   

            % Numerical method constants
            alpha = 0.0;
            obj.params.c0 = (1.5 + alpha) / (obj.dt * (1 + alpha));
            obj.params.c1 = -2 / obj.dt;
            obj.params.c2 = (0.5 + alpha) / (obj.dt * (1 + alpha));

            % Precompute constants
            obj.params.Kse_c0_Bse = obj.params.Kse + obj.params.c0 * obj.params.Bse;
            obj.params.Kbt_c0_Bbt = obj.params.Kbt + obj.params.c0 * obj.params.Bbt;
            obj.params.rho_A_g = obj.params.rho * obj.params.A * obj.params.g;

            % Get magnet parameters
            obj.params = mag_parameters(mag_set, obj.params);

            % Set transformation parameters
            obj.params.eul_t = [0.0, deg2rad(90), deg2rad(-90)];
            obj.params.H_t = eul2quat(obj.params.eul_t, "ZYX");
            obj.params.R_t = quat2rotm(obj.params.H_t);
            obj.params.T_t = [0.1; -0.85; 0.465]; % Translation vector
            obj.params.T_t = [0.0; 0.0; 0.0]; % Translation vector

            % Define the simulation setup for Python
            obj.params.matter = matter;
            obj.params.mag_set = mag_set;
        end

        function [L0, T0, magnets_length] = limits(obj)
            L0 = obj.s_length;
            T0 = sqrt(obj.params.rho * L0^2 / obj.params.E);
            magnets_length = obj.params.l_c + obj.params.l_e;
        end

        function [pos0, ang0, q0, w0, reset_terminated] = reset(obj, action0)
            p_e0 = action0(1:3);
            quat_e0 = action0(4:7);

            % Initialize history variables for Newmark integration
            obj.z = zeros(4, 6, obj.num_points);
            obj.z(3, 3, :) = 1; % Initial shear strain
            obj.z(3, 6, :) = 1; % Initial bending strain

            % Initialize state arrays
            obj.p_total = zeros(obj.num_points, 3, obj.steps + 1);
            obj.p_total(:, 3, 1) = obj.params.s(:); % Initial position along z-axis

            obj.h_total = zeros(obj.num_points, 4, obj.steps + 1);
            obj.h_total(:, :, 1) = repmat([1, 0, 0, 0], obj.num_points, 1); % Unit quaternion

            obj.q_total = zeros(obj.num_points, 3, obj.steps + 1);
            obj.w_total = zeros(obj.num_points, 3, obj.steps + 1);

            % Reset histories
            obj.p_e_history = zeros(obj.steps + 1, 3);
            obj.quat_e_history = zeros(obj.steps + 1, 4);

            obj.step_counter = 1;
            
            sol = [];
            t = 0;
            while t<50*obj.dt
            % while t<5*obj.dt
                [sol, p, h, n, m, q, w, obj.z, reset_terminated] = bvp_solver(p_e0, quat_e0, obj.z, obj.params, sol);
                t = t + obj.dt;
            end
            obj.sol_prev = sol;
            obj.sol_prev_rand = sol;
            obj.z_rand = obj.z;

            % Store initial tip state and initial action
            obj.p_total(:, :, 1) = p;
            obj.h_total(:, :, 1) = h;
            obj.q_total(:, :, 1) = q;
            obj.w_total(:, :, 1) = w;            
            obj.p_e_history(obj.step_counter, :) = p_e0';
            obj.quat_e_history(obj.step_counter, :) = quat_e0;

            % Return initial tip state in local coordinates
            pos0 = obj.p_total(end, :, 1);
            ang0 = quat2eul(obj.h_total(end, :, 1), "ZYX");
            q0 = obj.q_total(end, :, 1);
            w0 = obj.w_total(end, :, 1);
        end

        function [next_pos, next_ang, next_l_v, next_a_v, term] = step(obj, action)
            % Step: Advance simulation by one time step in local coordinates
            % Solve BVP for current external magnet position and orientation
            [obj.sol_prev, p, h, n, m, q, w, obj.z, term] = bvp_solver(action(1:3), action(4:7), obj.z, obj.params, obj.sol_prev);

            % Check for NaN or Inf values
            if any(isnan(p(:))) || any(isinf(p(:)))
                warning('NaN or Inf detected in position at step %d. Stopping simulation.', obj.step_counter);
            end

            % Prepare outputs in local coordinates
            next_pos = p(end, :);
            next_ang = quat2eul(h(end, :), "ZYX");
            next_l_v = q(end, :);
            next_a_v = w(end, :);
        end

        function [next_pos, next_ang, next_l_v, next_a_v, target_term] = rand_setpoint(obj, action)
            % Step: Advance simulation by one time step in local coordinates
            % Solve BVP for current external magnet position and orientation
            [sol, p, h, n, m, q, w, sol_z, target_term] = bvp_solver(action(1:3), action(4:7), obj.z_rand, obj.params, obj.sol_prev_rand);

            obj.step_counter = obj.step_counter + 1;

            % Only update the rand baseline solution if it wasn't a failure
            if ~target_term
                obj.sol_prev_rand = sol; 
                obj.z_rand = sol_z;
            end

            % Prepare outputs in local coordinates
            next_pos = p(end, :);
            next_ang = quat2eul(h(end, :), "ZYX");
            next_l_v = q(end, :);
            next_a_v = w(end, :);
        end

        function [next_pos, next_ang, next_l_v, next_a_v, term] = test_model(obj, action)
            % Step: Advance simulation by one time step in local coordinates
            % Solve BVP for current external magnet position and orientation
            [obj.sol_prev, p, h, n, m, q, w, obj.z, term] = bvp_solver(action(1:3), action(4:7), obj.z, obj.params, obj.sol_prev);
            
            % Check for NaN or Inf values
            if any(isnan(p(:))) || any(isinf(p(:)))
                warning('NaN or Inf detected in position at step %d. Stopping simulation.', obj.step_counter);
            end
            
            obj.step_counter = obj.step_counter + 1;

            % Store p_e and eul_e in history (locacoordinates)
            obj.p_e_history(obj.step_counter, :) = action(1:3);
            obj.quat_e_history(obj.step_counter, :) = action(4:7);

            % Update state in local coordinates
            obj.p_total(:, :, obj.step_counter) = p;
            obj.h_total(:, :, obj.step_counter) = h;
            obj.q_total(:, :, obj.step_counter) = q;
            obj.w_total(:, :, obj.step_counter) = w;

            % Prepare outputs in local coordinates
            next_pos = p(end, :);
            next_ang = quat2eul(h(end, :), "ZYX");
            next_l_v = q(end, :);
            next_a_v = w(end, :);
        end

        function create_video(obj, folder_dir, flag, shape, freq, amp)
        % function create_video(obj, video_dir)
            % Create a video of the 3D trajectory including robot pose
            % Convert local coordinates to global for plotting
            [p_global, h_global, p_e_global, quat_e_global] = local_global(obj.p_total, obj.h_total, obj.p_e_history, obj.quat_e_history, obj.params);

            % Prepare parameters for plotting
            parameters = obj.params;
            parameters.p_e = p_e_global;
            parameters.quat_e = quat_e_global;
            parameters.dt = obj.dt;

            parameters.flag = flag;
            parameters.shape = shape;
            parameters.freq = freq;
            parameters.amp = amp;

            % Call the plotting function with global coordinates
            plot_trajectory_as_vid(p_global, h_global, p_e_global, quat_e_global, parameters, folder_dir);
            % plot_trajectory_as_vid(p_global, h_global, p_e_global, quat_e_global, parameters, video_dir);
        end
    end
end

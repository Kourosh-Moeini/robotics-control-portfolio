classdef Simulation < handle
    properties
        num_points
        s_length
        ds
        dt
        t
        steps
        params
        p_total
        h_total
        sol_prev
        sol_prev_rand_point
        options_fsolve
        step_counter
    end
    
    methods
        function obj = Simulation(matter, mag_set)
            % Constructor: Initialize simulation parameters
            switch matter
                case 'Rod1'
                    obj.num_points = 20;
                    obj.s_length = 0.03;
                    obj.steps = 500;
                    obj.dt = 0.01;
                    t_length = obj.steps * obj.dt;
                    obj.t = (0:obj.dt:t_length)';
                    s = linspace(0, obj.s_length, obj.num_points);
                    obj.ds = s(2) - s(1);
                case 'Rod2'
                    obj.num_points = 50;
                    obj.s_length = 0.1;
                    obj.steps = 500;
                    obj.dt = 0.01;
                    t_length = obj.steps * obj.dt;
                    obj.t = (0:obj.dt:t_length)';
                    s = linspace(0, obj.s_length, obj.num_points);
                    obj.ds = s(2) - s(1);
                case 'PDMS1'
                    obj.num_points = 50;
                    obj.s_length = 0.10669;
                    obj.steps = 500;
                    obj.dt = 0.01;
                    t_length = obj.steps * obj.dt;
                    obj.t = (0:obj.dt:t_length)';
                    s = linspace(0, obj.s_length, obj.num_points);
                    obj.ds = s(2) - s(1);
                case 'PDMS2'
                    obj.num_points = 50;
                    obj.s_length = 0.12830 - 0.00855./2;
                    obj.steps = 500;
                    obj.dt = 0.01;
                    t_length = obj.steps * obj.dt;
                    obj.t = (0:obj.dt:t_length)';
                    s = linspace(0, obj.s_length, obj.num_points);
                    obj.ds = s(2) - s(1);

            end

            % Get material parameters
            obj.params = parameters(matter);
            obj.params.num_points = obj.num_points;
            obj.params.s_length = obj.s_length;
            obj.params.steps = obj.steps;
            obj.params.ds = obj.ds;
            obj.params.s = s;        

            % Get magnet parameters
            obj.params = mag_parameters(mag_set, obj.params);

            % Define the simulation setup for Python
            obj.params.matter = matter;
            obj.params.mag_set = mag_set;
            shape = 'Inf';
            obj.params.shape = shape;      

            % Set up fsolve options
            obj.options_fsolve = optimoptions('fsolve', 'Display', 'none', ...
                'FunctionTolerance', 1e-10, 'StepTolerance', 1e-10, 'OptimalityTolerance',...
                1e-10, 'UseParallel', false);

            % Initialize state
            obj.reset();
        end
        
        function [L0, F0, Tau0] = limits(obj)
            L0 = obj.s_length;
            F0 = 1.0;
            Tau0 = 0.001;
        end

        function [pos, ang] = reset(obj)
            % Reset: Initialize simulation state in local coordinates
            obj.p_total = zeros(obj.num_points, 3, obj.steps + 1);
            obj.p_total(:, 3, 1) = obj.params.s(:);  % Along z-axis in local frame
            
            obj.h_total = zeros(obj.num_points, 4, obj.steps + 1);
            obj.h_total(:, :, 1) = repmat([1, 0, 0, 0], obj.num_points, 1);  % Identity quaternion
            
            obj.step_counter = 1;
            % Previous solution for initial guess (empty at start)
            obj.sol_prev = [];
            obj.sol_prev_rand_point = [];

            % Return initial tip state in local coordinates
            pos = obj.p_total(end, :, 1);
            ang = quat2eul(obj.h_total(end, :, 1), 'ZYX');
        end
        
        function [F_ext, L_ext, next_pos, next_ang, done] = step(obj, Wrench_end)
            Force = Wrench_end(1:3)';
            Tau = Wrench_end(4:6)';
        
            % Solve BVP for this step
            [sol, p, h, n, m, done] = bvp_solver(Force, Tau, obj.params, obj.sol_prev);

            % Update previous solution for next guess
            obj.sol_prev = sol;
            
            % Update state in local coordinates
            obj.p_total(:, :, obj.step_counter + 1) = p;
            obj.h_total(:, :, obj.step_counter + 1) = h;
            
            % Prepare outputs in local coordinates
            next_pos = p(end, :);
            next_ang = quat2eul(h(end, :), 'ZYX');
            F_ext = n(end, :);
            L_ext = m(end, :);
            % done = obj.step_counter >= obj.steps;
            
            obj.step_counter = obj.step_counter + 1;
        end
        
        function [F_ext, L_ext, next_pos, next_ang, done] = rand_pos(obj, Wrench_end)
            Force = Wrench_end(1:3)';
            Tau = Wrench_end(4:6)';
        
            % Solve BVP for this step
            [sol, p, h, n, m, done] = bvp_solver(Force, Tau, obj.params, obj.sol_prev_rand_point);

            % Update previous solution for next guess
            obj.sol_prev_rand_point = sol;
            
            % Prepare outputs in local coordinates
            next_pos = p(end, :);
            next_ang = quat2eul(h(end, :), 'ZYX');
            F_ext = n(end, :);
            L_ext = m(end, :);
            % done = obj.step_counter >= obj.steps;
            done = 0;
        end

        function create_video(obj, folder_dir, flag, freq, amp)
        % function create_video(obj, video_dir)
            % Create a video of the 3D trajectory including robot pose            
            % Prepare parameters for plotting
            parameters = obj.params;

            parameters.flag = flag;
            parameters.freq = freq;
            parameters.amp = amp;
            
            % Call the plotting function with global coordinates
            plot_trajectory_as_vid(obj.p_total, obj.h_total, parameters, folder_dir);
            % plot_trajectory_as_vid(obj.p_total, obj.h_total, parameters, video_dir);
        end
    end
end

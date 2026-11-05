function [p, h, n, m] = objective_func(T0, Force, params)
    %#codegen
    % For initial guess in non-magnet case; improved with ode45
    p0 = [0; 0; 0];
    h0 = [1; 0; 0; 0];
    n0 = Force';
    m0 = T0';
    initial_state = [p0; h0; n0; m0];
    tspan = params.s;
    opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'Jacobian', @(s, y) odes_Jacobian(s, y, params)); % Tight for precision
    [~, states] = ode45(@(s, state) odes(s, state, params), tspan, initial_state, opts);

    % state = initial_state;
    % states = zeros(params.num_points, 13);
    % states(1, :) = initial_state';
    % 
    % for point_counter = 1:params.num_points-1
    %     K1 = odes(point_counter, state, params);
    %     K2 = odes(point_counter, state + params.ds*K1/2, params);
    %     K3 = odes(point_counter, state + params.ds*K2/2, params);
    %     K4 = odes(point_counter + 1, state + params.ds*K3, params);
    % 
    %     state = state + params.ds*(K1/6 + K2/3 + K3/3 + K4/6);
    %     state(4:7) = state(4:7)/norm(state(4:7));
    % 
    %     states(point_counter+1, :) = state';
    % end

    p = states(:, 1:3);
    h = states(:, 4:7);
    n = states(:, 8:10);
    m = states(:, 11:13);
end

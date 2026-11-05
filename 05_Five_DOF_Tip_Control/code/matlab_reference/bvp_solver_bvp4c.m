% file 3-bvp_solver.m (revised to use bvp4c with magnet boundary conditions)
function [sol, p, h, B_field, terminated] = bvp_solver(p_e_step, quat_e_step, params, sol_prev)
    % Options for bvp4c
    options_bvp = bvpset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Nmax', 500, ...
                         'FJacobian', @(s, y) odes_Jacobian(s, y, params), ...
                         'BCJacobian', @(ya, yb) bc_Jacobian(ya, yb, p_e_step, quat_e_step, params));

    % options_bvp = bvpset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Nmax', 500, ...
    %                      'FJacobian', @(s, y) odes_Jacobian(s, y, params));

    % options_bvp = bvpset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Nmax', 500);

    % Prepare initial guess for bvp4c
    s_span = params.s(:)';
    if isempty(sol_prev)
    % Initial guess (using objective_func with zero guesses for n0 and m0)
        m_guess = zeros(1, 3);
        Force_guess = zeros(1, 3); % Or use [0, 0, -params.mass_magnet * 9.81] if g is known, but keep zero for simplicity
        [p_g, h_g, n_g, m_g] = objective_func(m_guess, Force_guess, params);
        states_guess = [p_g, h_g, n_g, m_g]; % num_points x 13
        solinit.x = s_span;
        solinit.y = states_guess'; % 13 x num_points
    else
    % Use previous solution as guess for better convergence
        y_prev = deval(sol_prev, s_span);
        solinit.x = s_span;
        solinit.y = y_prev; % 13 x num_points
    end
    % Define odefun and bcfun for this step
    odefun_handle = @(s, y) odes(s, y, params);
    bcfun = @(ya, yb) bc_magnet(ya, yb, p_e_step, quat_e_step, params);

    % Solve BVP
    sol = bvp4c(odefun_handle, bcfun, solinit, options_bvp);

    % Check convergence (residual at boundaries)
    res = bcfun(sol.y(:, 1), sol.y(:, end));

    if norm(res) > 1e-5
        terminated = 1; % Failure
    else
        terminated = 0; % Converged
    end

    % Extract states at evaluation points
    y = deval(sol, s_span);
    p = y(1:3, :)';
    h = y(4:7, :)';

    R_tip = my_quat2rotm(h(end, :)');
    m_c = R_tip * params.m_c_local;
    R_e = quat2rotm(quat_e_step);
    m_e = R_e * params.m_e_local;
    [~, ~, B_field, ~] = force_torque(p(end, :)', p_e_step, m_e, m_c, params.mu0, R_e);

end

function res = bc_magnet(ya, yb, p_e_step, quat_e_step, params)
    % Boundary conditions with magnet interactions
    p_tip = yb(1:3);
    h_tip = yb(4:7);
    R_tip = my_quat2rotm(h_tip');
    m_c = R_tip * params.m_c_local;
    R_e = quat2rotm(quat_e_step);
    m_e = R_e * params.m_e_local;
    [F_magnetic, Tau_magnetic, ~, ~] = force_torque(p_tip', p_e_step, m_e, m_c, params.mu0, R_e);
    F_gravity = params.mass_magnet * params.g;
    F_total = (F_magnetic + F_gravity);
    Tau_total = Tau_magnetic;
    res = [ya(1:3) - [0; 0; 0]; ...
           ya(4:7) - [1; 0; 0; 0]; ...
           yb(8:10) - F_total(:); ...
           yb(11:13) - Tau_total(:)];
end

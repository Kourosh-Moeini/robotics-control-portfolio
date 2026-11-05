function err = objective_func(guess, params, interp_data)
    %#codegen
    % Extract base internal forces and moments from guess
    n0 = guess(1:3);
    m0 = guess(4:6);
    
    % Initial state vector: [p0; h0; n0; m0; q0; w0]
    y0 = [0; 0; 0; 1; 0; 0; 0; n0; m0; 0; 0; 0; 0; 0; 0];
    
    % Integrate using RK4
    Y = RK4(params, interp_data, y0, 4); % 4 substeps for accuracy
    
    % Extract tip state
    y_tip = Y(:, end);
    p_tip = y_tip(1:3);
    h_tip = y_tip(4:7);
    n_tip = y_tip(8:10);
    m_tip = y_tip(11:13);
    
    % Compute magnetic forces and torques at tip
    R_e = my_quat2rotm(interp_data.quat_e);
    m_e = R_e * interp_data.m_e_local; % External magnet moment in global frame
    
    R_tip = my_quat2rotm(h_tip);
    m_c = R_tip * interp_data.m_c_local; % Catheter magnet moment in global frame
    
    % Compute magnetic interaction
    [F_magnetic, Tau_magnetic] = force_torque(p_tip, interp_data.p_e, m_e, m_c, interp_data.mu0, R_e);
    
    % Add gravity force (assuming magnet is at tip)
    F_gravity = interp_data.mass_magnet * interp_data.g;
    F_total = F_magnetic + F_gravity;
    
    % Error: difference between internal forces/moments and applied loads
    err = [n_tip - F_total; m_tip - Tau_magnetic];
end

function R = my_quat2rotm(q)
%#codegen
% Convert quaternion to rotation matrix
q = q / norm(q);
w = q(1);
x = q(2);
y = q(3);
z = q(4);

x2 = x*x;
y2 = y*y;
z2 = z*z;

R = [1 - 2*(y2 + z2), 2*(x*y - w*z), 2*(x*z + w*y);
     2*(x*y + w*z), 1 - 2*(x2 + z2), 2*(y*z - w*x);
     2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x2 + y2)];
end

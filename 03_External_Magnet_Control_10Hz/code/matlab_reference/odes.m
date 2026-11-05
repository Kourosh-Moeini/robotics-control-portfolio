function dState = odes(s, state, params, interp_data)
    %#codegen
    % Extract state variables
    p = state(1:3);      % Position
    h = state(4:7);      % Orientation (quaternion)
    n = state(8:10);     % Internal force
    m = state(11:13);    % Internal moment
    q = state(14:16);    % Linear velocity
    w = state(17:19);    % Angular velocity
    
    % Convert quaternion to rotation matrix
    R = my_quat2rotm(h');
    
    % Interpolate history terms
    s_mesh = interp_data.s_mesh;
    ds = interp_data.ds;
    
    % Find interpolation index
    if s <= s_mesh(1)
        idx = 1;
        frac = 0;
    elseif s >= s_mesh(end)
        idx = length(s_mesh) - 1;
        frac = 1;
    else
        idx = max(1, min(floor((s - s_mesh(1)) / ds) + 1, length(s_mesh) - 1));
        frac = (s - s_mesh(idx)) / ds;
    end
    
    % Interpolate history terms
    qh = interp1d(idx, frac, interp_data.qh_array)';
    wh = interp1d(idx, frac, interp_data.wh_array)';
    vh = interp1d(idx, frac, interp_data.vh_array)';
    uh = interp1d(idx, frac, interp_data.uh_array)';
    
    % Compute strains using constitutive relations with damping
    v = params.Kse_c0_Bse \ (R' * n + params.Kse * params.v0 - params.Bse * vh);
    u = params.Kbt_c0_Bbt \ (R' * m + params.Kbt * params.u0 - params.Bbt * uh);
    
    % Construct skew-symmetric matrix for quaternion derivative
    U = [0, -u(1), -u(2), -u(3);
         u(1), 0, u(3), -u(2);
         u(2), -u(3), 0, u(1);
         u(3), u(2), -u(1), 0];
    
    % Compute external forces (distributed gravity)
    f = params.rho_A_g - R * params.C * q .* abs(q) + params.f_bar;
    l = params.l_bar;
    
    % Compute time derivatives using Newmark integration
    dv_dt = params.c0 * v + vh;
    du_dt = params.c0 * u + uh;
    dq_dt = params.c0 * q + qh;
    dw_dt = params.c0 * w + wh;
    
    % Spatial derivatives
    dP_ds = R * v;
    dh_ds = 0.5 * U * h;
    dn_ds = R * (params.rho * params.A * (cross(w, q) + dq_dt)) - f;
    dm_ds = R * (cross(w, params.rho*params.J*w) + params.rho*params.J*dw_dt) - cross(dP_ds, n) - l;
    dq_ds = dv_dt - cross(u, q) + cross(w, v);
    dw_ds = du_dt - cross(u, w);
    
    % Combine all derivatives
    dState = [dP_ds; dh_ds; dn_ds; dm_ds; dq_ds; dw_ds];
end

function val = interp1d(idx, frac, array)
    %#codegen
    % Manual linear interpolation for code generation compatibility
    val = array(idx, :) + frac * (array(idx+1, :) - array(idx, :));
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

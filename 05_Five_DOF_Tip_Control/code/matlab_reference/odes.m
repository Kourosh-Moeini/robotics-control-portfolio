function dState = odes(s, state, params)
    %#codegen
    p = state(1:3);
    h = state(4:7)/norm(state(4:7));
    R = my_quat2rotm(h');
    n = state(8:10);
    m = state(11:13);

    % Compute v and u
    v = params.Kse \ (R' * n) + params.v0;
    u = params.Kbt \ (R' * m) + params.u0;

    % Compute U matrix
    U = [0, -u(1), -u(2), -u(3);
         u(1), 0, u(3), -u(2);
         u(2), -u(3), 0, u(1);
         u(3), u(2), -u(1), 0];

    % Derivatives with distributed gravity in global frame
    dP_ds = R * v;
    dh_ds = 0.5 * U * h;
    f_global = params.rho * params.A * params.g;  % Distributed gravity
    dn_ds = -f_global;
    dm_ds = -cross(dP_ds, n);

    dState = [dP_ds; dh_ds; dn_ds; dm_ds];
end

% function R = my_quat2rotm(q)
% %#codegen
% % A codegen-compatible function to convert a quaternion to a rotation matrix.
% % Input q is a 4x1 column vector [w; x; y; z].
% 
%     % Ensure the quaternion is normalized
%     q = q / norm(q);
% 
%     w = q(1);
%     x = q(2);
%     y = q(3);
%     z = q(4);
% 
%     % Pre-calculate squared terms
%     x2 = x*x;
%     y2 = y*y;
%     z2 = z*z;
% 
%     % Calculate rotation matrix elements
%     R = [1 - 2*(y2 + z2),   2*(x*y - w*z),     2*(x*z + w*y);
%          2*(x*y + w*z),     1 - 2*(x2 + z2),   2*(y*z - w*x);
%          2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x2 + y2)];
% end

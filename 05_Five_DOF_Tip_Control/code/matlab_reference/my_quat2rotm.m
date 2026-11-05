function R = my_quat2rotm(q)
%#codegen
% A codegen-compatible function to convert a quaternion to a rotation matrix.
% Input q is a 4x1 column vector [w; x; y; z].

    % Ensure the quaternion is normalized
    q = q / norm(q);

    w = q(1);
    x = q(2);
    y = q(3);
    z = q(4);
    
    % Pre-calculate squared terms
    x2 = x*x;
    y2 = y*y;
    z2 = z*z;
    
    % Calculate rotation matrix elements
    R = [1 - 2*(y2 + z2),   2*(x*y - w*z),     2*(x*z + w*y);
         2*(x*y + w*z),     1 - 2*(x2 + z2),   2*(y*z - w*x);
         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x2 + y2)];
end

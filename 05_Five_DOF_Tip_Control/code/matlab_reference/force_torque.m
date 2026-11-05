function [f_mag, tau_mag, B_global, J_local] = force_torque(p_tip, p_e, m_e, m_c, mu0, R_e)
    % FORCE_TORQUE Exact analytical magnetic force and torque.
    % Computes BOTH the B-field and its spatial gradient analytically
    % inside a single loop, providing extreme precision and high speed.
    
    % 1. Ensure all inputs are column vectors
    p_tip = p_tip(:);
    p_e   = p_e(:);
    m_e   = m_e(:);
    m_c   = m_c(:);
    
    % 2. Define External Magnet Dimensions
    dim = [0.05; 0.06; 0.06]; 
    vol = dim(1) * dim(2) * dim(3);
    
    % 3. Calculate Exact Signed Remanence (Br) in Local Frame
    m_e_local = R_e' * m_e; 
    Br_z = mu0 * (m_e_local(3) / vol); 
    
    if abs(Br_z) < 1e-12
        f_mag = [0; 0; 0];
        tau_mag = [0; 0; 0];
        return;
    end
    
    % 4. Transform tip position into local magnet frame
    p_local = R_e' * (p_tip - p_e); 
    
    % 5. Get Analytical B-field and Exact Analytical Jacobian (Gradient)
    [B_local, J_local] = cuboid_analytical_B_and_Grad(p_local, dim, Br_z);
    
    % 6. Rotate B-field and Jacobian back to Global Frame
    B_global = R_e * B_local;
    
    % 7. Calculate Magnetic Torque: tau = m_c x B
    tau_mag = cross(m_c, B_global);
    
    % 8. Calculate Magnetic Force using the analytical Jacobian
    % Transform catheter moment to local frame, compute force, and rotate back
    m_c_local = R_e' * m_c;
    f_local = J_local * m_c_local;   % F = (m_c * nabla) B inside local frame
    f_mag = R_e * f_local;           % Transform force to global frame
end

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function [B, J] = cuboid_analytical_B_and_Grad(p_local, dim, Br_z)
    % Calculates the exact analytical B-field AND its 3x3 spatial gradient 
    % (Jacobian tensor) for a Z-magnetized cuboid in a single pass.
    
    X = p_local(1); Y = p_local(2); Z = p_local(3);
    
    a = dim(1) / 2; 
    b = dim(2) / 2; 
    c = dim(3) / 2;
    
    U = [X - a, X + a];
    V = [Y - b, Y + b];
    W = [Z - c, Z + c];
    
    % Initialize B-field components
    Bx = 0; By = 0; Bz = 0;
    
    % Initialize independent Jacobian components
    Jxx = 0; Jxy = 0; Jxz = 0;
    Jyy = 0; Jyz = 0; 
    
    % Surface integration limits over the 8 vertices
    for i = 1:2
        for j = 1:2
            for k = 1:2
                u = U(i); 
                v = V(j); 
                w = W(k);
                
                r = sqrt(u^2 + v^2 + w^2);
                
                if r < 1e-9
                    continue; % Protect against singularity at exact corners
                end
                
                sgn = (-1)^(i + j + k);
                
                % --- 1. Analytical B-Field ---
                Bx = Bx + sgn * log(v + r);
                By = By + sgn * log(u + r);
                % Bz = Bz - sgn * atan2(u * v, w * r);
                
                % Replace atan2(Y, X) with this logic:
                Y = u * v;
                X = w * r;

                % Standard real atan2
                real_part = atan2(real(Y), real(X)); 

                % Derivative stored in imaginary part
                imag_part = (real(X)*imag(Y) - real(Y)*imag(X)) / (real(X)^2 + real(Y)^2); 

                % Reconstruct the complex number
                Bz = Bz - sgn * complex(real_part, imag_part);

                % --- 2. Analytical Spatial Gradients ---
                % Max is used to prevent divide-by-zero if evaluated 
                % perfectly on the extended axes where (v+r)=0 or (u+r)=0.
                denom_v = max(r * (v + r), 1e-12);
                denom_u = max(r * (u + r), 1e-12);
                
                Jxx = Jxx + sgn * (u / denom_v);
                Jyy = Jyy + sgn * (v / denom_u);
                Jxy = Jxy + sgn * (1 / r);
                Jxz = Jxz + sgn * (w / denom_v);
                Jyz = Jyz + sgn * (w / denom_u);
            end
        end
    end
    
    coef = Br_z / (4 * pi);
    B = [coef * Bx; coef * By; coef * Bz];
    
    % Derive remaining Jacobian components using Maxwell's equations:
    % 1. Divergence-free (div B = 0) -> dZ/dz = -(dX/dx + dY/dy)
    Jzz = -(Jxx + Jyy); 
    
    % 2. Curl-free outside magnet (curl B = 0) -> Symmetric tensor
    Jyx = Jxy;
    Jzx = Jxz;
    Jzy = Jyz;

    % Construct the 3x3 Gradient Tensor
    J = coef * [Jxx, Jxy, Jxz;
                Jyx, Jyy, Jyz;
                Jzx, Jzy, Jzz];
end

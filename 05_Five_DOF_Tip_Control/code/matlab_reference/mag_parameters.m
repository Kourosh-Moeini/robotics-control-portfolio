function params = mag_parameters(mag_set, params)
    % Define magnet properties based on set
    switch mag_set
        case 'Set1'
            % Catheter magnet properties
            params.r_c = 0.001;         % radius (m)
            params.l_c = 0.005;         % height (m)
            M_c = 8e6;                  % magnetization (A/m)
            density_magnet = 7500;      % density (kg/m³)

            % External magnet properties
            params.r_e = 0.01;          % radius (m)
            params.l_e = 0.02;          % height (m)
            M_e = 8e6;                  % magnetization (A/m)

            V_c = pi * params.r_c^2 * params.l_c;
            V_e = pi * params.r_e^2 * params.l_e;

        case 'Set2'
            % Catheter magnet properties
            params.r_c = 0.0024650;         % radius (m)
            params.l_c = 0.00855;         % height (m)
            M_c = 6.75e5;                % magnetization (A/m)
            density_magnet = 1.1e-3./(params.l_c*pi*params.r_c^2);      % density (kg/m³)

            % External magnet properties
            params.r_e = 0.025;         % radius (m)
            params.l_e = 0.05;         % height (m)
            M_e = 6.8e5;                % magnetization (A/m)

            V_c = pi * params.r_c^2 * params.l_c;
            V_e = pi * params.r_e^2 * params.l_e;

        case 'Set3'
            % Catheter magnet properties
            params.r_c = 0.0024650;         % radius (m)
            params.l_c = 0.00855;         % height (m)
            M_c = 6.62549e5;                % magnetization (A/m)
            density_magnet = 1.1e-3./(params.l_c*pi*params.r_c^2);      % density (kg/m³)

            % External magnet properties
            params.r_e = 0.025;         % radius (m)
            params.l_e = 0.06;         % height (m)
            M_e = 1.055130e6;                % magnetization (A/m)

            V_c = pi * params.r_c^2 * params.l_c;
            V_e = 0.06 * 0.05 * 0.06;

        otherwise
            error('Unknown magnet set: %s', mag_set);
    end

    % Compute catheter magnet properties
    params.m_c_local = [0; 0; -V_c * M_c];
    params.mass_magnet = density_magnet * V_c;

    % Compute external magnet local magnetic moment
    params.m_e_local = [0; 0; -V_e * M_e];

    % Physical constant
    params.mu0 = 4 * pi * 1e-7;  % Permeability of free space (H/m)
end

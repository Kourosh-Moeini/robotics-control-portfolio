function params = parameters(matter)
    params = struct();
    switch matter
        case 'Rod1'
            params.rho = 907;
            r = 0.005;
            r0 = 0.003;
            params.nu = 0.3;
            params.E = 8000e3;
            G = params.E / (2 * (1 + params.nu));
            params.g = [-9.81; 0.0; 0.0];
        case 'Rod2'
            params.rho = 8000;
            r = 0.0024 / 2;
            r0 = 0;
            params.nu = 0.3;
            params.E = 207 * 10^9;
            G = params.E / (2 * (1 + params.nu));
            params.g = [0.0; -9.81; 0.0];
        case 'PDMS1'
            params.rho = 970;
            r = (5.0e-3)/2;
            r0 = (3.0e-3)/2;
            params.nu = 0.478000878687799;
            params.E = 3.347591925420653e+05;
            G = params.E / (2 * (1 + params.nu));
            params.g = [0.0; 0.0; 0.0];
        case 'PDMS2'
           params.rho = 1030;
            r = (6.77e-3)/2;
            r0 = (4.930e-3)/2;
            % params.nu = 0.422429111252843;
            % params.E = 1.888567124751886e+06;
            params.nu = 0.45;
            params.E = 1.05e+06;
            G = params.E / (2 * (1 + params.nu));
            params.g = [0.0; 0.0; 0.0];
            % params.g = [9.81; 0.0; 0.0];

    end

    %% Geometric Properties
    params.A = pi * (r^2 - r0^2);

    %% Second Moment of Area
    params.J = zeros(3, 3);
    params.J(1, 1) = pi * (r^4 - r0^4) / 4;
    params.J(2, 2) = pi * (r^4 - r0^4) / 4;
    params.J(3, 3) = pi * (r^4 - r0^4) / 2;

    %% Stiffness Matrices
    params.Kse = zeros(3, 3);
    params.Kse(1, 1) = G * params.A;
    params.Kse(2, 2) = G * params.A;
    params.Kse(3, 3) = params.E * params.A;

    params.Kbt = zeros(3, 3);
    params.Kbt(1, 1) = params.E * params.J(1, 1);
    params.Kbt(2, 2) = params.E * params.J(2, 2);
    params.Kbt(3, 3) = G * params.J(3, 3);

    %% Initial Conditions
    params.v0 = [0; 0; 1];
    params.u0 = [0; 0; 0];

    %% External Forces and Moments
    params.f_bar = [0; 0; 0];
    params.l_bar = [0; 0; 0];
end

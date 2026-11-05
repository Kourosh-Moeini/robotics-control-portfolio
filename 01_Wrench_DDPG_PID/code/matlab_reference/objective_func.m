function err = objective_func(Force_target, Tau_target, guess, params)
    %#codegen
    n0 = guess(1:3);
    m0 = guess(4:6);
    y0 = [0; 0; 0; 1; 0; 0; 0; n0; m0];  % Base: p=0, h=unit, n=n0, m=m0
    tspan = params.s;  % Assuming params.s is row vector of increasing points
    [~, Y] = ode45(@(s, y) odes(s, y, params), tspan, y0);  % Y is num_points x 13
    yb = Y(end, :);

    F_gravity = params.mass_magnet * params.g;
    Force_total = Force_target + F_gravity;
    err = [yb(8:10)' - Force_total(:); yb(11:13)' - Tau_target(:)];
end

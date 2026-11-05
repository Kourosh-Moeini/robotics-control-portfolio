function Y = RK4(params, interp_data, y0, substeps)
    %#codegen
    % RK4 integrator with substeps per mesh interval
    N = length(params.s);
    dim = length(y0);
    Y = zeros(dim, N);
    Y(:, 1) = y0;
    
    % Step size for substeps
    h_sub = params.ds / substeps;
    
    for i = 1:N-1
        y = Y(:, i);
        s = params.s(i);
        
        % RK4 with substeps
        for k = 1:substeps
            k1 = odes(s, y, params, interp_data);
            k2 = odes(s + h_sub/2, y + (h_sub/2) * k1, params, interp_data);
            k3 = odes(s + h_sub/2, y + (h_sub/2) * k2, params, interp_data);
            k4 = odes(s + h_sub, y + h_sub * k3, params, interp_data);
            
            y = y + (h_sub/6) * (k1 + 2*k2 + 2*k3 + k4);
            s = s + h_sub;
        end
        
        Y(:, i+1) = y;
    end
end

function poldersm(K_goal)
% Run the simulation using the goal maintenance level K_goal
% i.e. polders(0.9)
%
% Goals used in this study: 0.5, 0.9, 1.2

    % *Initial Model Setup*
    years = 100;

    % Initial values %
    W      = 0;          % Initial water level
    F      = 0;          % Initial flood intensity
    H      = 0;          % No damage initially
    M      = 1;          % Initial value in Yu et al., 2017
    S      = 2.5;        % Soil salinity in dS/m (deciSiemens per meter)
    K      = K_goal;     % Set initial K at goal
    X      = 0.95;       % C = X, D = 1 - X
    C = 1; D = 0;        % For checking values later
    P   = 0.5;             % Social capital level

    % For plotting later %
    W_track = zeros(1,years);
    K_track = zeros(1,years);
    S_track = zeros(1,years);
    M_track = zeros(1,years);
    X_track = zeros(1,years);
    P_track = zeros(1,years);
    pi_C_track = zeros(1,years);
    pi_D_track = zeros(1,years);
    pi_bar_track = zeros(1,years);

    % Setup population matrix
    % (1)C/D-1/0   (2)l_f    (3)l_m    (4)pi (current)    (5)Pi (aggregate)
    size = 5875;
    population = zeros(size, 5);
    % Set distribution of C/D
    for n = 1:size
        if rand() < X % initial C/D ratio
           population(n, 1) = C;
        end
    end

    l_total = 12; % total available labor per household
    L = l_total * size; L_f = 0; L_m = 0;

    % Parameter values %
    delta_K = 0.08;% erosion factor
    sigma_F = 1; % damage sensitivity
    a = 3;  % acreage is constant and the same for everyone
    b0 = 0.1311; % initial productivity factor
    psi = 1;
    b = b0*psi; % productivity factor in farming
    alpha = 0.7; % output elasticity in farming
    z = 0.03; % test parameter (0.03-0.06) wage rate in external
    j = 0.0004; % productivity factor in maintenance
    beta = 0.6; % output elasticity in maintenance
    lambda = 0.02; % default sanctioning cost (test parameter)
    epsilon = 0.5; % relative increase per unit increase in 1-X
    gamma = 0.01; % default punishment
    xi_X = 1.6; % relative increase per unit X
    xi_M = 0.8; % relative increase per unit M
    eta = 10; % sensitivity to change
    mu_M = 1; % rate of gain of social memory
    delta_M = 0.05; % rate of decay of memory
    mu_S = 12.5; % rate of gain in soil salinity
    delta_S = 0.4; % leaching efficiency
    ECen = 3; % Lower threshold of soil salinity above which the stress affects crop production
    ECex = 11.3; % Upper threshold above which stress is so severe that crop production ceases
    sigma_V = 5; % degree by which breach risk escalates
    zeta = 0.13; % coeff translates high water level to breach level

    % *Simulation*

    rng default % for reproducibility

    for year = 1:years

    % Set up normal distribution for water height level %
    W = max(normrnd(0.70,0.18),0);

    % Calculate flood intensity %
    if W > K
        F = 1 - exp(-sigma_F * W);

        % Calculate breach potential and damage %
        V = 1 - exp(-sigma_V * (W-K));
        if rand() < V
            H = zeta * W;
        else
            H = 0;
        end
    else
        F = 0;
        H = 0;
        V = 0;
    end

    % Calculate soil salinity change %
    dSdt = mu_S * F - delta_S * S;
    S = max(S + dSdt,0);

    % Calculate social memory change %
    dMdt = mu_M * F - delta_M * M;
    M = max(M + dMdt,0);

    % Calculate psi %
    if S <= ECen
        psi = 1;
    elseif S >= ECex
        psi = 0;
    else
        psi = 1 - (S-ECen) / (ECex-ECen);
    end

    % Caclulate farming productivity %
    b = b0 * psi;

    % Calculate L_m from K* %
    K_off = max(K_goal - K + delta_K * K,0);
    L_m_goal = ((K_off / j) * (1 / P^(1-beta)))^(1/beta);
    l_m_goal = (1 / size) * L_m_goal;
    l_m_C = l_m_goal;
    l_m_D = 0;

    % Maximize contributor / defector income and assign to population %
    % Use fmincon: https://www.mathworks.com/help/optim/ug/fmincon.html %
    c_func = @(l) -b*l(1)^alpha*a^(1-alpha) - z*l(2);
    c_Aeq = [1,1]; c_beq = max(l_total - l_m_C,0);
    c_lb = [0,0]; c_ub = [max(l_total-l_m_C,0), max(l_total-l_m_C,0)];
    l_C_0 = [1,1];
    l_C = fmincon(c_func, l_C_0, [], [], c_Aeq, c_beq, c_lb, c_ub);
    l_f_C = l_C(1);
    l_e_C = l_C(2);

    d_func = @(l) -b*l(1)^alpha*a^(1-alpha) + z*l(2);
    d_Aeq = [1,1]; d_beq = l_total;
    d_lb = [0,0]; d_ub = [l_total, l_total];
    l_D_0 = [1,1];
    l_D = fmincon(d_func, l_D_0, [], [], d_Aeq, d_beq, d_lb, d_ub);
    l_f_D = l_D(1);
    l_e_D = l_D(2);

    pi_C = (1+F)*(b*l_f_C^alpha*a^(1-alpha)+z*l_e_C) - lambda*(1+epsilon*(1-X));
    pi_D = (1-F)*(b*l_f_D^alpha*a^(1-alpha)-z*l_e_D) - gamma*(1+xi_X*X)*(1+xi_M*M);

    % Assign maximized incomes %
    for n = 1:size
        if population(n, 1) == C
            population(n, 2) = l_f_C;
            population(n, 3) = l_m_C;
            population(n, 4) = pi_C;
            population(n, 5) = population(n, 5) + pi_C;
        elseif population(n, 1) == D
            population(n, 2) = l_f_D;
            population(n, 3) = l_m_D;
            population(n, 4) = pi_D;
            population(n, 5) = population(n, 5) + pi_D;
        end
    end

    % And calculate aggregates %
    % For use in rule determination %
    L_f = sum(population(:,2));
    L_m = sum(population(:,3));

    % Check overall social capital %
    P = max(sum(population(:,5)), 0);
    P = (P / size) / year;

    % Check who's changing sides %
    pi_bar = sum(population(:,4)) / size; % avg income
    dXdt = X * (eta * (pi_C - pi_bar));
    X = min(max(X + dXdt, 0), 1);

    % And asssign new population ratios %
    for n = 1:size
        if rand() < X
            population(n, 1) = C;
        else
            population(n, 1) = D;
        end
    end

    % Finally, calculate change in levee height %
    R = j * L_m^beta * P^(1-beta);
    dKdt = R - delta_K*K - H;
    K = max(K + dKdt,0);

    W_track(year) = W;
    K_track(year) = K;
    S_track(year) = S;
    M_track(year) = M;
    X_track(year) = X;
    P_track(year) = P;
    pi_C_track(year) = pi_C;
    pi_D_track(year) = pi_D;
    pi_bar_track(year) = pi_bar;
    end

    % *Plotting and Data Analysis*

    plot1 = figure;
    hold on
    plot(1:years, K_track);
    bar(W_track, 0.3, 'r');
    axis([0 years 0 1.4]);
    legend('K', 'W');
    ylabel('Height');
    title(['K* = ' num2str(K_goal)]);
    hold off

    plot2 = figure;
    hold on
    plot(1:years, S_track, 1:years, P_track, 1:years, M_track);
    legend('S', 'P', 'M');
    hold off

    plot3 = figure;
    hold on
    plot(1:years, X_track);
    legend('X');
    hold off

    plot4 = figure;
    hold on
    plot(1:years, pi_C_track, 1:years, pi_D_track);
    legend('pi_C', 'pi_D');
    hold off

    plot5 = figure;
    hold on
    plot(1:years, pi_C_track, 1:years, pi_D_track, 1:years, pi_bar_track);
    legend('pi_C', 'pi_D', 'pi_{avg}');
    xlabel('Year');
    hold off
end
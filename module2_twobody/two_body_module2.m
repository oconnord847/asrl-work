% Two-Body Problem: Euler vs RK4 vs ASSET
% Module 2 | Declan O'Connor | ASRL
%
% REQUIRES: asset_orbit.csv in the same folder as this script.
% Run asset_twobody_earth.py in WSL first, then copy the CSV next to this file.

clear; clc; close all;

%% Constants
G   = 6.67430e-20;       % km^3 kg^-1 s^-2
m_E = 5.97219e24;        % kg
mu  = G * m_E;
R_E = 6378.12;           % km

%% Initial Conditions
r0 = [8000; 0; 6000];    % km
v0 = [0; 7; 0];          % km/s
Y0 = [r0; v0];

%% Time vector
t0 = 0; tf = 14709; dt = 10;
t  = (t0:dt:tf)';
N  = length(t);

%% Euler Integration
Y_euler = zeros(6, N);
Y_euler(:,1) = Y0;
for i = 1:N-1
    Y_euler(:,i+1) = Y_euler(:,i) + dt * twobody_ode(Y_euler(:,i), mu);
end

%% RK4 Integration
Y_rk4 = zeros(6, N);
Y_rk4(:,1) = Y0;
for i = 1:N-1
    k1 = twobody_ode(Y_rk4(:,i),              mu);
    k2 = twobody_ode(Y_rk4(:,i) + (dt/2)*k1, mu);
    k3 = twobody_ode(Y_rk4(:,i) + (dt/2)*k2, mu);
    k4 = twobody_ode(Y_rk4(:,i) + dt*k3,     mu);
    Y_rk4(:,i+1) = Y_rk4(:,i) + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

%% Load ASSET Data
A = readmatrix('asset_orbit.csv', 'NumHeaderLines', 1);
t_asset = A(:,1);
x_a = A(:,2); y_a = A(:,3); z_a = A(:,4);

%% Interpolate ASSET onto RK4 time grid
x_ai = interp1(t_asset, x_a, t, 'pchip');
y_ai = interp1(t_asset, y_a, t, 'pchip');
z_ai = interp1(t_asset, z_a, t, 'pchip');

%% Component and total position deltas: RK4 vs ASSET (km)
dx = Y_rk4(1,:)' - x_ai;
dy = Y_rk4(2,:)' - y_ai;
dz = Y_rk4(3,:)' - z_ai;
delta_total = sqrt(dx.^2 + dy.^2 + dz.^2);

[peak_val, ip] = max(delta_total);
peak_hr  = t(ip)/3600;
mean_val = mean(delta_total);

%% Altitudes
alt_euler = vecnorm(Y_euler(1:3,:)) - R_E;
alt_rk4   = vecnorm(Y_rk4(1:3,:))  - R_E;
alt_asset = sqrt(x_ai.^2 + y_ai.^2 + z_ai.^2) - R_E;

fprintf('=== Altitude Summary (km) ===\n')
fprintf('Euler   min %.2f   max %.2f\n', min(alt_euler), max(alt_euler))
fprintf('RK4     min %.2f   max %.2f\n', min(alt_rk4),   max(alt_rk4))
fprintf('ASSET   min %.2f   max %.2f\n', min(alt_asset),  max(alt_asset))
fprintf('=== RK4 vs ASSET Position Delta (km) ===\n')
fprintf('Peak  %.4f km at %.2f hr\n', peak_val, peak_hr)
fprintf('Mean  %.4f km\n', mean_val)

%% FIGURE 1: 3D Orbit Comparison
fig1 = figure('Name','3D Orbit','Position',[50 50 800 650],'Color','w');
set(fig1, 'InvertHardcopy', 'off')
ax1 = axes(fig1);
hold(ax1, 'on')

p_e = plot3(ax1, Y_euler(1,:), Y_euler(2,:), Y_euler(3,:), 'r-', 'LineWidth', 1.2);
p_r = plot3(ax1, Y_rk4(1,:), Y_rk4(2,:), Y_rk4(3,:), 'Color', [0 0.5 0], 'LineWidth', 1.6);
p_a = plot3(ax1, x_ai, y_ai, z_ai, 'b--', 'LineWidth', 1.6);

[sx, sy, sz] = sphere(40);
surf(ax1, sx*R_E, sy*R_E, sz*R_E, 'FaceColor', [0.3 0.55 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5)
p_0 = plot3(ax1, r0(1), r0(2), r0(3), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

styleAxes(ax1)
xlabel(ax1, 'X (km)'); ylabel(ax1, 'Y (km)'); zlabel(ax1, 'Z (km)')
title(ax1, 'Two-Body Orbit: Euler vs RK4 vs ASSET')
legend(ax1, [p_e p_r p_a p_0], {'Euler','RK4','ASSET','r_0'}, 'Location','best','TextColor','k')
axis(ax1,'equal'); view(ax1, 35, 22)

%% FIGURE 2: The Delta Figure (two stacked panels)
fig2 = figure('Name','RK4 vs ASSET Delta','Position',[880 50 950 720],'Color','w');
set(fig2, 'InvertHardcopy', 'off')

% Top panel: altitude overlay
axT = subplot(2,1,1, 'Parent', fig2);
hold(axT, 'on')
plot(axT, t/3600, alt_rk4,   'Color',[0 0.5 0], 'LineWidth', 1.6);
plot(axT, t/3600, alt_asset, 'b--', 'LineWidth', 1.6);
styleAxes(axT)
xlabel(axT, 'Time (hours)'); ylabel(axT, 'Altitude (km)')
title(axT, 'Altitude vs Time: RK4 and ASSET appear identical')
legend(axT, {'RK4','ASSET'}, 'Location','best','TextColor','k')
xlim(axT, [0 t(end)/3600])

% Bottom panel: position delta, total + components
axB = subplot(2,1,2, 'Parent', fig2);
hold(axB, 'on')
h_tot = plot(axB, t/3600, delta_total, 'b-',  'LineWidth', 2.0);
h_dx  = plot(axB, t/3600, abs(dx),     'r-',  'LineWidth', 1.0);
h_dy  = plot(axB, t/3600, abs(dy),     'Color',[0 0.5 0], 'LineWidth', 1.0);
h_dz  = plot(axB, t/3600, abs(dz),     'Color',[0.85 0.5 0], 'LineWidth', 1.0);

plot(axB, peak_hr, peak_val, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k')
text(axB, peak_hr, peak_val, sprintf('  peak %.2f km', peak_val), ...
    'Color','k','FontWeight','bold','VerticalAlignment','bottom')

styleAxes(axB)
xlabel(axB, 'Time (hours)'); ylabel(axB, 'Position difference (km)')
title(axB, 'RK4 vs ASSET: absolute position difference, broken into components')
legend(axB, [h_tot h_dx h_dy h_dz], {'Total |\Deltar|','|\Deltax|','|\Deltay|','|\Deltaz|'}, ...
    'Location','best','TextColor','k')
xlim(axB, [0 t(end)/3600])

annotation(fig2, 'textbox', [0.15 0.16 0.30 0.10], ...
    'String', {sprintf('Peak: %.2f km', peak_val), sprintf('Mean: %.2f km', mean_val)}, ...
    'Color','k', 'EdgeColor',[0.6 0.6 0.6], 'BackgroundColor','w', ...
    'FontWeight','bold', 'FitBoxToText','on')

%% FIGURE 3: Altitude vs Time (all three)
fig3 = figure('Name','Altitude vs Time','Position',[50 730 950 400],'Color','w');
set(fig3, 'InvertHardcopy', 'off')
ax3 = axes(fig3);
hold(ax3, 'on')
plot(ax3, t/3600, alt_euler, 'r-', 'LineWidth', 1.2);
plot(ax3, t/3600, alt_rk4,   'Color',[0 0.5 0], 'LineWidth', 1.6);
plot(ax3, t/3600, alt_asset, 'b--', 'LineWidth', 1.6);
styleAxes(ax3)
xlabel(ax3, 'Time (hours)'); ylabel(ax3, 'Altitude (km)')
title(ax3, 'Altitude vs Time: Euler vs RK4 vs ASSET')
legend(ax3, {'Euler','RK4','ASSET'}, 'Location','best','TextColor','k')

%% Styling helper: white axes, black text, grid
function styleAxes(ax)
    set(ax, 'Color','w', 'XColor','k', 'YColor','k', 'ZColor','k', ...
        'GridColor',[0.7 0.7 0.7], 'FontSize', 11, 'FontName', 'Arial')
    grid(ax, 'on')
    set(get(ax,'Title'),  'Color','k', 'FontSize', 13, 'FontWeight','bold')
    set(get(ax,'XLabel'), 'Color','k', 'FontSize', 12)
    set(get(ax,'YLabel'), 'Color','k', 'FontSize', 12)
    set(get(ax,'ZLabel'), 'Color','k', 'FontSize', 12)
end

%% ODE Function
function dY = twobody_ode(Y, mu)
    r = Y(1:3);
    v = Y(4:6);
    a = -mu * r / norm(r)^3;
    dY = [v; a];
end

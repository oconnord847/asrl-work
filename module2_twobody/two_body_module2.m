% Two-Body Problem: Euler vs RK4 vs ASSET
% Module 2 | Declan O'Connor | ASRL
%
% REQUIRES: asset_orbit.csv in the same folder as this script.
% Run asset_twobody_earth.py in WSL first, then:
%   cp ~/asrl_work/asset_orbit.csv /mnt/c/Users/oconn/Downloads/
% and place it next to this file.

clear; clc; close all;

%% ── Constants ─────────────────────────────────────────────────────────────────
G   = 6.67430e-20;       % km^3 kg^-1 s^-2
m_E = 5.97219e24;        % kg
mu  = G * m_E;
R_E = 6378.12;           % km

%% ── Initial Conditions ────────────────────────────────────────────────────────
r0 = [8000; 0; 6000];    % km
v0 = [0; 7; 0];          % km/s
Y0 = [r0; v0];

%% ── Time vector ───────────────────────────────────────────────────────────────
t0 = 0; tf = 14709; dt = 10;
t  = (t0:dt:tf)';        % column vector
N  = length(t);

%% ── Euler Integration ────────────────────────────────────────────────────────
Y_euler = zeros(6, N);
Y_euler(:,1) = Y0;
for i = 1:N-1
    Y_euler(:,i+1) = Y_euler(:,i) + dt * twobody_ode(Y_euler(:,i), mu);
end

%% ── RK4 Integration ──────────────────────────────────────────────────────────
Y_rk4 = zeros(6, N);
Y_rk4(:,1) = Y0;
for i = 1:N-1
    k1 = twobody_ode(Y_rk4(:,i),              mu);
    k2 = twobody_ode(Y_rk4(:,i) + (dt/2)*k1, mu);
    k3 = twobody_ode(Y_rk4(:,i) + (dt/2)*k2, mu);
    k4 = twobody_ode(Y_rk4(:,i) + dt*k3,     mu);
    Y_rk4(:,i+1) = Y_rk4(:,i) + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

%% ── Load ASSET Data ──────────────────────────────────────────────────────────
A = readmatrix('asset_orbit.csv', 'NumHeaderLines', 1);
t_asset = A(:,1);         % s
x_a     = A(:,2);         % km
y_a     = A(:,3);
z_a     = A(:,4);

%% ── Interpolate ASSET onto RK4 time grid ─────────────────────────────────────
% ASSET uses adaptive steps; interpolate to same t grid as RK4/Euler
x_ai = interp1(t_asset, x_a, t, 'pchip');
y_ai = interp1(t_asset, y_a, t, 'pchip');
z_ai = interp1(t_asset, z_a, t, 'pchip');

%% ── Compute Delta: |RK4 - ASSET| position error ─────────────────────────────
dx = Y_rk4(1,:)' - x_ai;
dy = Y_rk4(2,:)' - y_ai;
dz = Y_rk4(3,:)' - z_ai;
delta_km = sqrt(dx.^2 + dy.^2 + dz.^2);   % km

%% ── Altitudes ────────────────────────────────────────────────────────────────
alt_euler = vecnorm(Y_euler(1:3,:)) - R_E;
alt_rk4   = vecnorm(Y_rk4(1:3,:))  - R_E;
alt_asset = sqrt(x_ai.^2 + y_ai.^2 + z_ai.^2) - R_E;

fprintf('=== Altitude Summary ===\n')
fprintf('        Min Alt (km)   Max Alt (km)\n')
fprintf('Euler   %12.2f  %12.2f\n', min(alt_euler), max(alt_euler))
fprintf('RK4     %12.2f  %12.2f\n', min(alt_rk4),   max(alt_rk4))
fprintf('ASSET   %12.2f  %12.2f\n', min(alt_asset),  max(alt_asset))
fprintf('\n=== RK4 vs ASSET Position Delta ===\n')
fprintf('Max delta : %.4f km\n', max(delta_km))
fprintf('Mean delta: %.4f km\n', mean(delta_km))

%% ══════════════════════════════════════════════════════════════════════════════
%% FIGURE 1: 3D Orbit Comparison
%% ══════════════════════════════════════════════════════════════════════════════
fig1 = figure('Name','3D Orbit','Position',[50 50 800 650],'Color','w');
ax1 = axes(fig1);
set(ax1, 'Color','w', 'XColor','k', 'YColor','k', 'ZColor','k', ...
    'FontSize', 11, 'FontName', 'Arial')

hold(ax1, 'on')
plot3(ax1, Y_euler(1,:), Y_euler(2,:), Y_euler(3,:), ...
    'r-', 'LineWidth', 1.2, 'DisplayName', 'Euler')
plot3(ax1, Y_rk4(1,:),   Y_rk4(2,:),   Y_rk4(3,:), ...
    'Color', [0.0 0.6 0.0], 'LineWidth', 1.5, 'DisplayName', 'RK4')
plot3(ax1, x_ai, y_ai, z_ai, ...
    'b--', 'LineWidth', 1.5, 'DisplayName', 'ASSET')

% Earth sphere
[sx, sy, sz] = sphere(40);
surf(ax1, sx*R_E, sy*R_E, sz*R_E, ...
    'FaceColor', [0.2 0.5 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
    'HandleVisibility', 'off')

% Mark initial position
plot3(ax1, r0(1), r0(2), r0(3), 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k', ...
    'DisplayName', 'r_0')

xlabel(ax1, 'X (km)', 'FontSize', 12, 'Color', 'k')
ylabel(ax1, 'Y (km)', 'FontSize', 12, 'Color', 'k')
zlabel(ax1, 'Z (km)', 'FontSize', 12, 'Color', 'k')
title(ax1, 'Two-Body Orbit: Euler vs RK4 vs ASSET', 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', 'k')
legend(ax1, 'Location', 'best', 'FontSize', 10, 'TextColor', 'k', 'Color', 'w')
grid(ax1, 'on')
axis(ax1, 'equal')
view(ax1, 35, 22)

%% ══════════════════════════════════════════════════════════════════════════════
%% FIGURE 2: RK4 vs ASSET Position Delta (the important one)
%% ══════════════════════════════════════════════════════════════════════════════
fig2 = figure('Name','RK4 vs ASSET Delta','Position',[900 50 900 500],'Color','w');
ax2 = axes(fig2);
set(ax2, 'Color','w', 'XColor','k', 'YColor','k', 'FontSize', 11, ...
    'FontName', 'Arial')

plot(ax2, t/3600, delta_km, 'b-', 'LineWidth', 1.5)
hold(ax2, 'on')
yline(ax2, max(delta_km), 'r--', 'LineWidth', 1.2, ...
    'Label', sprintf('Peak: %.2f km', max(delta_km)), ...
    'LabelHorizontalAlignment', 'right', 'FontSize', 10)

% Mark the peak
[~, ip] = max(delta_km);
plot(ax2, t(ip)/3600, delta_km(ip), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
    'HandleVisibility', 'off')
text(ax2, t(ip)/3600 + 0.05, delta_km(ip) + max(delta_km)*0.04, ...
    sprintf('%.2f km', delta_km(ip)), ...
    'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold')

xlabel(ax2, 'Time (hours)', 'FontSize', 12, 'Color', 'k')
ylabel(ax2, 'Position Error (km)', 'FontSize', 12, 'Color', 'k')
title(ax2, {'RK4 vs ASSET: Absolute Position Difference', ...
    '\fontsize{10}\color{gray}Interpolated to common time grid | same initial conditions'}, ...
    'FontSize', 13, 'FontWeight', 'bold', 'Color', 'k')
grid(ax2, 'on')
xlim(ax2, [0, t(end)/3600])
ylim(ax2, [0, max(delta_km) * 1.2])

% Annotation box explaining the scale context
annotation(fig2, 'textbox', [0.13 0.68 0.45 0.13], ...
    'String', {'Scale context: what looks like a flat zero', ...
               'on an orbit plot can be hundreds of km here.'}, ...
    'FontSize', 9, 'Color', [0.3 0.3 0.3], 'EdgeColor', [0.8 0.8 0.8], ...
    'BackgroundColor', [0.97 0.97 0.97], 'FitBoxToText', 'on')

%% ══════════════════════════════════════════════════════════════════════════════
%% FIGURE 3: Altitude vs Time (all three methods)
%% ══════════════════════════════════════════════════════════════════════════════
fig3 = figure('Name','Altitude vs Time','Position',[50 750 900 400],'Color','w');
ax3 = axes(fig3);
set(ax3, 'Color','w', 'XColor','k', 'YColor','k', 'FontSize', 11, ...
    'FontName', 'Arial')

plot(ax3, t/3600, alt_euler, 'r-',  'LineWidth', 1.2, 'DisplayName', 'Euler')
hold(ax3, 'on')
plot(ax3, t/3600, alt_rk4,   'Color', [0.0 0.6 0.0], 'LineWidth', 1.5, ...
    'DisplayName', 'RK4')
plot(ax3, t/3600, alt_asset,  'b--', 'LineWidth', 1.5, 'DisplayName', 'ASSET')

xlabel(ax3, 'Time (hours)', 'FontSize', 12, 'Color', 'k')
ylabel(ax3, 'Altitude (km)', 'FontSize', 12, 'Color', 'k')
title(ax3, 'Altitude vs Time: Euler vs RK4 vs ASSET', 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', 'k')
legend(ax3, 'Location', 'best', 'FontSize', 10, 'TextColor', 'k', 'Color', 'w')
grid(ax3, 'on')

%% ── ODE Function ─────────────────────────────────────────────────────────────
function dY = twobody_ode(Y, mu)
    r = Y(1:3);
    v = Y(4:6);
    r_mag = norm(r);
    a     = -mu * r / r_mag^3;
    dY    = [v; a];
end

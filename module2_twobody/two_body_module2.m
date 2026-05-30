% Two-Body Problem: RK4 vs ASSET position delta (km), per-hour panels + scale
% Module 2 | Declan O'Connor | ASRL
% REQUIRES: asset_orbit.csv in the same folder as this script.

clear; clc; close all;

%% Constants
G = 6.67430e-20; m_E = 5.97219e24; mu = G * m_E; R_E = 6378.12;

%% Initial Conditions
r0 = [8000; 0; 6000]; v0 = [0; 7; 0]; Y0 = [r0; v0];

%% Time vector
t0 = 0; tf = 14709; dt = 10;
t = (t0:dt:tf)'; N = length(t);

%% RK4
Y_r = zeros(6,N); Y_r(:,1) = Y0;
for i = 1:N-1
    k1 = twobody_ode(Y_r(:,i),            mu);
    k2 = twobody_ode(Y_r(:,i)+(dt/2)*k1, mu);
    k3 = twobody_ode(Y_r(:,i)+(dt/2)*k2, mu);
    k4 = twobody_ode(Y_r(:,i)+dt*k3,     mu);
    Y_r(:,i+1) = Y_r(:,i) + (dt/6)*(k1+2*k2+2*k3+k4);
end

%% Load + interpolate ASSET onto RK4 time grid
A = readmatrix('asset_orbit.csv', 'NumHeaderLines', 1);
t_a = A(:,1); x_a = A(:,2); y_a = A(:,3); z_a = A(:,4);
x_ai = interp1(t_a, x_a, t, 'pchip');
y_ai = interp1(t_a, y_a, t, 'pchip');
z_ai = interp1(t_a, z_a, t, 'pchip');

%% Position delta in km
dx = Y_r(1,:)' - x_ai;
dy = Y_r(2,:)' - y_ai;
dz = Y_r(3,:)' - z_ai;
delta  = sqrt(dx.^2 + dy.^2 + dz.^2);   % km
t_hr   = t/3600;
r_orbit = vecnorm(Y_r(1:3,:))';          % orbit radius (km)

rawColor = [0 0.30 0.75];                % clear deep blue

%% FIGURE 1: 2x2 zoom panels at integer hours 1,2,3,4
hours = [1 2 3 4]; half = 0.25;
summary = zeros(numel(hours),5);   % hour, peak_km, peak_m, r_km, ppm

fig1 = figure('Name','Delta per hour','Position',[150 120 1050 720],'Color','w');
set(fig1, 'InvertHardcopy', 'off')
for k = 1:numel(hours)
    h = hours(k);
    w = t_hr >= h-half & t_hr <= h+half;
    idx = find(w);
    [pv, j] = max(delta(w)); ip = idx(j);
    summary(k,:) = [h, pv, pv*1000, r_orbit(ip), pv/r_orbit(ip)*1e6];

    ax = subplot(2,2,k,'Parent',fig1); hold(ax,'on')
    plot(ax, t_hr(w), delta(w), '-', 'Color', rawColor, 'LineWidth', 1.4);
    plot(ax, t_hr(ip), pv, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k')
    text(ax, t_hr(ip), pv, sprintf('  peak %.0f m', pv*1000), ...
        'Color','k','FontWeight','bold','VerticalAlignment','bottom')
    styleAxes(ax)
    xlabel(ax,'Time (hours)'); ylabel(ax,'|\Deltar| (km)')
    title(ax, sprintf('t = %d hr', h))
    xlim(ax, [h-half h+half]); ylim(ax, [0 pv*1.3])
end
sgtitle(fig1, 'RK4 vs ASSET Position Difference, by Hour', ...
    'FontWeight','bold','FontSize',14,'Color','k')

%% Summary table (proof) -> CSV + command window
S = table(summary(:,1), summary(:,2), summary(:,3), summary(:,4), summary(:,5), ...
    'VariableNames', {'hour','peak_km','peak_m','orbit_radius_km','peak_ppm_of_radius'});
writetable(S, 'delta_summary_by_hour.csv');
fprintf('=== Peak delta by hour ===\n'); disp(S)
fprintf('Table written: delta_summary_by_hour.csv\n');

%% FIGURE 2: scale representation (orbit radius vs peak delta, log axis)
peak_all = max(summary(:,2));        % km
r_max    = max(r_orbit);             % km
fig2 = figure('Name','Scale','Position',[250 150 760 320],'Color','w');
set(fig2, 'InvertHardcopy', 'off')
ax2 = axes(fig2);
b = barh(ax2, [1 2], [r_max, peak_all], 0.5);
b.FaceColor = 'flat'; b.CData(1,:) = [0.4 0.55 0.85]; b.CData(2,:) = rawColor;
set(ax2, 'XScale', 'log', 'YTick', [1 2], ...
    'YTickLabel', {'Max orbit radius','Peak |\Deltar|'})
styleAxes(ax2)
xlabel(ax2, 'Distance (km, log scale)')
title(ax2, sprintf('Scale: peak difference is %.0f ppm of the orbit radius', ...
    peak_all/r_max*1e6))
text(ax2, r_max,   1, sprintf(' %.0f km', r_max),   'Color','k','VerticalAlignment','middle')
text(ax2, peak_all,2, sprintf(' %.4f km (%.0f m)', peak_all, peak_all*1000), ...
    'Color','k','VerticalAlignment','middle')

%% Helpers
function styleAxes(ax)
    set(ax, 'Color','w', 'XColor','k', 'YColor','k', ...
        'GridColor',[0.7 0.7 0.7], 'FontSize', 11, 'FontName', 'Arial')
    grid(ax, 'on')
    set(get(ax,'Title'),  'Color','k', 'FontSize', 12, 'FontWeight','bold')
    set(get(ax,'XLabel'), 'Color','k', 'FontSize', 11)
    set(get(ax,'YLabel'), 'Color','k', 'FontSize', 11)
end
function dY = twobody_ode(Y, mu)
    r = Y(1:3); v = Y(4:6);
    dY = [v; -mu*r/norm(r)^3];
end

% CR3BP L1 Lyapunov Orbit: canonical frame + orbit zoom + to-scale frame
% Module 3 | Declan O'Connor | ASRL
% REQUIRES: cr3bp_orbit.csv in the same folder as this script.

clear; clc; close all;

%% Constants
mu = 0.01215;
lstar = 384400.0; R_earth = 6378.0; R_moon = 1737.4;
Re = R_earth/lstar; Rm = R_moon/lstar;
xE = -mu; xM = 1 - mu;

%% Load ASSET trajectory
A = readmatrix('cr3bp_orbit.csv', 'NumHeaderLines', 1);
x = A(:,2); y = A(:,3); z = A(:,4);
vx = A(:,5); vy = A(:,6); vz = A(:,7);

%% Lagrange points (Earth-Moon)
L1 = 0.8369; L2 = 1.1557; L3 = -1.0051;
L4x = 0.5 - mu; L4y =  sqrt(3)/2;
L5x = 0.5 - mu; L5y = -sqrt(3)/2;

%% Clearances + Jacobi
r1 = sqrt((x-xE).^2 + y.^2 + z.^2);
r2 = sqrt((x-xM).^2 + y.^2 + z.^2);
altE = min(r1)*lstar - R_earth; altM = min(r2)*lstar - R_moon;
U = 0.5*(x.^2 + y.^2) + (1-mu)./r1 + mu./r2;
C = 2*U - (vx.^2 + vy.^2 + vz.^2);
fprintf('Closest Earth: %.0f km   Closest Moon: %.0f km\n', altE, altM);
fprintf('Jacobi C mean %.4f, drift %.3e\n', mean(C), max(C)-min(C));

%% FIGURE
fig = figure('Name','CR3BP L1 Lyapunov','Position',[100 120 1500 520],'Color','w');
set(fig,'InvertHardcopy','off')

% Panel 1: canonical CR3BP frame
ax1 = subplot(1,3,1,'Parent',fig); hold(ax1,'on')
plot(ax1, x, y, 'r-', 'LineWidth', 1.3, 'DisplayName','L1 Lyapunov orbit')
plot(ax1, xE, 0, 'o','MarkerFaceColor',[0.10 0.30 0.90],'MarkerEdgeColor','k','MarkerSize',13,'DisplayName','Earth')
plot(ax1, xM, 0, 'o','MarkerFaceColor',[0.55 0.55 0.55],'MarkerEdgeColor','k','MarkerSize',8,'DisplayName','Moon')
plot(ax1, 0,0,'g+','MarkerSize',10,'LineWidth',1.3,'DisplayName','Barycenter')
plot(ax1, L1,0,'*','Color',[0.85 0.10 0.10],'MarkerSize',10,'DisplayName','L1')
plot(ax1, L2,0,'*','Color',[0.10 0.65 0.10],'MarkerSize',10,'DisplayName','L2')
plot(ax1, L3,0,'*','Color',[0.10 0.20 0.85],'MarkerSize',10,'DisplayName','L3')
plot(ax1, L4x,L4y,'*','Color',[0.85 0.10 0.85],'MarkerSize',10,'DisplayName','L4')
plot(ax1, L5x,L5y,'*','Color',[0.10 0.75 0.85],'MarkerSize',10,'DisplayName','L5')
styleAxes(ax1); axis(ax1,'equal')
xlabel(ax1,'x (nondim)'); ylabel(ax1,'y (nondim)')
title(ax1,'CR3BP Rotating Frame')
Lg = legend(ax1,'Location','eastoutside'); set(Lg,'Color','w','TextColor','k','EdgeColor',[0.6 0.6 0.6])

% Panel 2: zoom on the orbit around L1
ax2 = subplot(1,3,2,'Parent',fig); hold(ax2,'on')
plot(ax2, x, y, 'r-', 'LineWidth', 1.6)
plot(ax2, L1, 0, '*','Color',[0.85 0.10 0.10],'MarkerSize',12)
text(ax2, L1, 0, '  L1','Color','k','VerticalAlignment','top')
styleAxes(ax2); axis(ax2,'equal')
padx = 0.1*(max(x)-min(x)); pady = 0.15*(max(y)-min(y));
xlim(ax2,[min(x)-padx max(x)+padx]); ylim(ax2,[min(y)-pady max(y)+pady])
xlabel(ax2,'x (nondim)'); ylabel(ax2,'y (nondim)')
title(ax2,'Zoom: the orbit about L1')

% Panel 3: full frame with Earth and Moon drawn TO SCALE
ax3 = subplot(1,3,3,'Parent',fig); hold(ax3,'on')
plot(ax3, x, y, 'r-', 'LineWidth', 1.3)
drawBody(ax3, xE, 0, Re, [0.10 0.30 0.90])
drawBody(ax3, xM, 0, Rm, [0.55 0.55 0.55])
plot(ax3, L1,0,'k.','MarkerSize',8)
text(ax3, xE,0,' Earth','Color','k','VerticalAlignment','top')
text(ax3, xM,0,' Moon','Color','k','VerticalAlignment','top')
styleAxes(ax3); axis(ax3,'equal')
xlim(ax3,[-0.15 1.05]); ylim(ax3,[-0.2 0.2])
xlabel(ax3,'x (nondim)'); ylabel(ax3,'y (nondim)')
title(ax3,'Bodies drawn to true scale')

sgtitle(fig, sprintf('CR3BP L1 Lyapunov Orbit  |  Jacobi C = %.4f  |  clears Moon by %.0f km', ...
    mean(C), altM), 'FontWeight','bold','FontSize',13,'Color','k')

%% Helpers
function drawBody(ax, cx, cy, R, col)
    th = linspace(0,2*pi,200);
    fill(ax, cx+R*cos(th), cy+R*sin(th), col, 'EdgeColor','none')
end
function styleAxes(ax)
    set(ax,'Color','w','XColor','k','YColor','k','GridColor',[0.75 0.75 0.75], ...
        'FontSize',10,'FontName','Arial'); grid(ax,'on')
    set(get(ax,'Title'),'Color','k','FontSize',12,'FontWeight','bold')
    set(get(ax,'XLabel'),'Color','k','FontSize',11)
    set(get(ax,'YLabel'),'Color','k','FontSize',11)
end

% CR3BP L1 & L2 Lyapunov Families: 3D perspective showpiece
% Module 3/4 | Declan O'Connor | ASRL
% REQUIRES: cr3bp_families.csv in the same folder as this script.

clear; clc; close all;

%% Constants
mu = 0.01215;
lstar = 384400.0; R_earth = 6378.0; R_moon = 1737.4;
xE = -mu; xM = 1 - mu;
L1 = 0.8369; L2 = 1.15568;

%% Load families
A = readmatrix('cr3bp_families.csv', 'NumHeaderLines', 1);
fam = A(:,1); oid = A(:,2); x = A(:,4); y = A(:,5); z = A(:,6);

%% Figure: dark background (no stars)
fig = figure('Name','Lyapunov & DRO Families','Position',[60 60 1300 800],'Color','k');
set(fig,'InvertHardcopy','off')
ax = axes(fig); hold(ax,'on'); set(ax,'Color','k')

%% Color gradients: L1 cool (winter), L2 warm (autumn), DRO (cool: cyan->magenta)
c1 = winter(5); c2 = autumn(5); c3 = cool(5);

for k = 0:4
    m = fam==0 & oid==k;
    plot3(ax, x(m), y(m), z(m), '-', 'Color', c1(k+1,:), 'LineWidth', 1.6)
end
for k = 0:4
    m = fam==1 & oid==k;
    plot3(ax, x(m), y(m), z(m), '-', 'Color', c2(k+1,:), 'LineWidth', 1.6)
end
for k = 0:4
    m = fam==2 & oid==k;
    plot3(ax, x(m), y(m), z(m), '-', 'Color', c3(k+1,:), 'LineWidth', 1.6)
end

%% Bodies (enlarged for visibility, as in reference diagrams)
drawSphere(ax, xE, 0, 0, 0.055, [0.20 0.50 0.95])   % Earth
drawSphere(ax, xM, 0, 0, 0.030, [0.75 0.75 0.78])   % Moon

%% Lagrange labels
text(ax, L1, 0, 0.07, 'L_1', 'Color','w','FontSize',20,'FontWeight','bold','FontName','Times')
text(ax, L2, 0, 0.07, 'L_2', 'Color','w','FontSize',20,'FontWeight','bold','FontName','Times')
text(ax, xE, 0.10, 0, 'Earth', 'Color','w','FontSize',12)
text(ax, xM, -0.07, 0, 'Moon', 'Color','w','FontSize',11)
text(ax, 0.95, 0.27, 0, 'Lyapunov orbits', 'Color','w','FontSize',13,'FontAngle','italic')
text(ax, xM, 0.16, 0, 'DROs', 'Color',[0.7 0.9 1],'FontSize',12,'FontAngle','italic')

%% Dimension lines
dimLine(ax, xE, xM, -0.20, sprintf('%s km','384,400'))   % Earth-Moon
dimLine(ax, L1, L2,  0.33, sprintf('%s km','122,533'))   % L1-L2

%% View / styling
axis(ax,'equal'); axis(ax,'off')
view(ax, -60, 26)
set(ax,'Projection','perspective')
xlim(ax,[-0.15 1.30]); ylim(ax,[-0.35 0.40]); zlim(ax,[-0.25 0.25])
title(ax, 'CR3BP Earth-Moon: L_1, L_2 Lyapunov Families and Lunar DROs', ...
    'Color','w','FontSize',15,'FontWeight','bold')

%% Helpers
function drawSphere(ax, cx, cy, cz, R, col)
    [sx,sy,sz] = sphere(24);
    surf(ax, cx+R*sx, cy+R*sy, cz+R*sz, 'FaceColor',col, 'EdgeColor','none', ...
        'FaceLighting','gouraud','AmbientStrength',0.6)
end
function dimLine(ax, xa, xb, yoff, label)
    plot3(ax, [xa xb], [yoff yoff], [0 0], '-', 'Color',[0.3 0.7 1], 'LineWidth',1.2)
    for xx = [xa xb]
        plot3(ax, [xx xx], [yoff-0.015 yoff+0.015], [0 0], '-', 'Color',[0.3 0.7 1], 'LineWidth',1.2)
    end
    text(ax, (xa+xb)/2, yoff-0.03, 0, label, 'Color','w','FontSize',12, ...
        'HorizontalAlignment','center','FontWeight','bold')
end

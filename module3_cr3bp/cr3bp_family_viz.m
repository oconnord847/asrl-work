% CR3BP L1, L2 Lyapunov + DRO families: 2D rotating-frame views
% Module 3/4 | Declan O'Connor | ASRL
% REQUIRES: cr3bp_families.csv in the same folder as this script.

clear; clc; close all;

%% Constants
mu = 0.01215;
lstar = 384400.0; R_earth = 6378.0; R_moon = 1737.4;
Re = R_earth/lstar; Rm = R_moon/lstar;
xE = -mu; xM = 1 - mu;
L1 = 0.8369; L2 = 1.15568; L3 = -1.0051;
L4x = 0.5-mu; L4y = sqrt(3)/2; L5x = 0.5-mu; L5y = -sqrt(3)/2;

%% Load families
A = readmatrix('cr3bp_families.csv', 'NumHeaderLines', 1);
fam = A(:,1); oid = A(:,2); x = A(:,4); y = A(:,5);

c1 = winter(5); c2 = autumn(5); c3 = cool(5);

%% ===== FIGURE 1: all three families, rotating frame, to scale =====
f1 = figure('Name','Families (rotating frame)','Position',[80 70 980 820],'Color','w');
set(f1,'InvertHardcopy','off'); ax = axes(f1); hold(ax,'on')
h1=[];h2=[];h3=[];
for k=0:4
    m=fam==0&oid==k; h1=plot(ax,x(m),y(m),'-','Color',c1(k+1,:),'LineWidth',1.5);
end
for k=0:4
    m=fam==1&oid==k; h2=plot(ax,x(m),y(m),'-','Color',c2(k+1,:),'LineWidth',1.5);
end
for k=0:4
    m=fam==2&oid==k; h3=plot(ax,x(m),y(m),'-','Color',c3(k+1,:),'LineWidth',1.5);
end
drawBody(ax,xE,0,Re,[0.12 0.32 0.85]); drawBody(ax,xM,0,Rm,[0.55 0.55 0.55])
plot(ax,0,0,'g+','MarkerSize',9,'LineWidth',1.2)
plot(ax,L1,0,'k*','MarkerSize',8); text(ax,L1,0,' L1','Color','k','VerticalAlignment','bottom')
plot(ax,L2,0,'k*','MarkerSize',8); text(ax,L2,0,' L2','Color','k','VerticalAlignment','bottom')
text(ax,xE,-Re,' Earth','Color','k','VerticalAlignment','top','FontWeight','bold')
text(ax,xM,-Rm,' Moon','Color','k','VerticalAlignment','top','FontWeight','bold')
styleAxes(ax); axis(ax,'equal')
xlim(ax,[-0.12 1.28]); ylim(ax,[-0.30 0.30])
xlabel(ax,'x (nondim)'); ylabel(ax,'y (nondim)')
title(ax,'L1, L2 Lyapunov Families and Lunar DROs')
Lg=legend(ax,[h1 h2 h3],{'L1 Lyapunov','L2 Lyapunov','DRO (around Moon)'}, ...
    'Location','southwest'); set(Lg,'Color','w','TextColor','k','EdgeColor',[0.6 0.6 0.6])

%% ===== FIGURE 2: Lyapunov spacing measurement (like before) =====
f2 = figure('Name','Lyapunov spacing','Position',[220 70 860 820],'Color','w');
set(f2,'InvertHardcopy','off'); ax2 = axes(f2); hold(ax2,'on')
apex=zeros(5,2);
for k=0:4
    m=fam==0&oid==k; xk=x(m); yk=y(m);
    plot(ax2,xk,yk,'-','Color',c1(k+1,:),'LineWidth',1.8);
    [yt,it]=max(yk); apex(k+1,:)=[xk(it),yt];
end
drawBody(ax2,xM,0,Rm,[0.55 0.55 0.55])
plot(ax2,L1,0,'r*','MarkerSize',11); text(ax2,L1,0,'  L1','Color','k','VerticalAlignment','top')
[~,ord]=sort(apex(:,2));
for j=1:4
    p1=apex(ord(j),:); p2=apex(ord(j+1),:);
    plot(ax2,[p1(1) p2(1)],[p1(2) p2(2)],'k--','LineWidth',1.0)
    plot(ax2,p1(1),p1(2),'k.','MarkerSize',10); plot(ax2,p2(1),p2(2),'k.','MarkerSize',10)
    mid=(p1+p2)/2; d=norm(p2-p1)*lstar;
    text(ax2,mid(1)+0.004,mid(2),sprintf('%.0f km',d),'Color','k','FontWeight','bold','FontSize',9)
end
styleAxes(ax2); axis(ax2,'equal')
xlim(ax2,[0.78 1.00]); ylim(ax2,[-0.25 0.27])
xlabel(ax2,'x (nondim)'); ylabel(ax2,'y (nondim)')
title(ax2,'L1 Lyapunov family: spacing between adjacent orbits')

%% Helpers
function drawBody(ax,cx,cy,R,col)
    th=linspace(0,2*pi,200); fill(ax,cx+R*cos(th),cy+R*sin(th),col,'EdgeColor','none','HandleVisibility','off')
end
function styleAxes(ax)
    set(ax,'Color','w','XColor','k','YColor','k','GridColor',[0.78 0.78 0.78],'FontSize',11,'FontName','Arial'); grid(ax,'on')
    set(get(ax,'Title'),'Color','k','FontSize',13,'FontWeight','bold')
    set(get(ax,'XLabel'),'Color','k','FontSize',12); set(get(ax,'YLabel'),'Color','k','FontSize',12)
end

% ========================================================================
% MATLAB Script: Publication-Ready Figures
% Professional style, exact PDF size, high resolution
% ========================================================================

close all; clear; clc;

% Set default font
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);

% -------------------------------
% 1. Loss History Plot
% -------------------------------
loss_file = dir('Semi2D_loss_data.mat');
if ~isempty(loss_file)
    load(loss_file(1).name);  % loss_history, term1_history, term2_history, term3_history, bc_history

    fig = figure('Units','inches','Position',[0 0 4.3 2]); % figure size
    hold on;
    plot(loss_history, 'Color',[0 0.45 0.74],'LineWidth',1);
    plot(term1_history, 'Color',[0.85 0.33 0.10],'LineWidth',1, 'LineStyle', '--');
    plot(bc_history, 'Color',[0.47 0.67 0.19],'LineWidth',1, 'LineStyle', '--');
    set(gca,'FontName','Times New Roman');
    xlabel('Epoch'); ylabel('Loss');
    set(gca,'YScale','log');
    grid off;
    box on
    legend({'Total','|Energy|^2','|BC|^2'}, 'Location', 'northeast', 'NumColumns', 3);
    box on;

    % Save PDF with exact figure size
    set(fig,'PaperUnits','inches');
    set(fig,'PaperPositionMode','auto');
%     print(fig,'Semi2D_loss.pdf','-dpdf','-r600');
    exportgraphics(fig, sprintf('Semi2D_loss.pdf'), 'ContentType', 'vector', 'Resolution', 600);
    close(fig);
end
%%
% -------------------------------
% 2. F Surface and Derivatives
% -------------------------------
fixed_points = [0.5 0.5; 0.25 0.75; 0.75 0.25];

for i = 1:size(fixed_points,1)
    x_val = fixed_points(i,1);
    y_val = fixed_points(i,2);

    % Load F
    files = dir(sprintf('Semi2D_3d_F_x*%g*_y*%g*_data.mat', x_val, y_val));
    if isempty(files)
        warning('F file not found for x=%g, y=%g', x_val, y_val);
        continue
    end
    load(files(1).name); % Z, W, F

    % ----- F surface -----
    fig = figure('Units','inches','Position',[0 0 2.5 1.9]);
    surf(Z, W, F, 'EdgeColor','none'); 
    colormap(cividis); shading interp;
    camlight headlight; lighting phong;
    xlabel('\xi'); ylabel('\tau'); zlabel('F');
    view(30,10);
%     view(60,30);
    grid on;
    set(gca,'FontName','Times New Roman');
    set(fig,'PaperUnits','inches'); set(fig,'PaperPositionMode','auto');
    box on;
    
    % print(fig,sprintf('Semi2D_3d_F_x%g_y%g.pdf', x_val, y_val),'-dpdf','-r600');
    exportgraphics(fig, sprintf('Semi2D_3d_F_x%g_y%g.pdf', x_val, y_val), 'ContentType', 'vector', 'Resolution', 600);

    close(fig);

    % ----- dF/dz surface -----
    files = dir(sprintf('Semi2D_3d_dFdz_x*%g*_y*%g*_data.mat', x_val, y_val));
    if isempty(files)
        warning('dFdz file not found for x=%g, y=%g', x_val, y_val);
        continue
    end
    load(files(1).name); % dFdz

    fig = figure('Units','inches','Position',[0 0 2.5 1.9]);
    surf(Z, W, dFdz, 'EdgeColor','none'); 
    colormap(summer); shading interp;
    camlight headlight; lighting phong;
    xlabel('\xi'); ylabel('\tau'); zlabel('\partial F/\partial \xi');
%     view(30,10);
    view(120,25);
    grid on; set(gca,'FontName','Times New Roman');
    set(fig,'PaperUnits','inches'); set(fig,'PaperPositionMode','auto');
    box on;
    zlim([-1.1, 1.1])
    % print(fig,sprintf('Semi2D_3d_dFdz_x%g_y%g.pdf', x_val, y_val),'-dpdf','-r600');
    exportgraphics(fig, sprintf('Semi2D_3d_dFdz_x%g_y%g.pdf', x_val, y_val), 'ContentType', 'vector', 'Resolution', 600);

    close(fig);

    % ----- dF/dw surface -----
    files = dir(sprintf('Semi2D_3d_dFdw_x*%g*_y*%g*_data.mat', x_val, y_val));
    if isempty(files)
        warning('dFdw file not found for x=%g, y=%g', x_val, y_val);
        continue
    end
    load(files(1).name); % dFdw

    fig = figure('Units','inches','Position',[0 0 2.5 1.9]);
    surf(Z, W, dFdw, 'EdgeColor','none'); 
    colormap(autumn); shading interp;
    camlight headlight; lighting phong;
    xlabel('\xi'); ylabel('\tau'); zlabel('\partial F/\partial \tau');
%     view(30,10);
    view(120, 25);
    grid on; set(gca,'FontName','Times New Roman');
    set(fig,'PaperUnits','inches'); set(fig,'PaperPositionMode','auto');
    box on;
    zlim([-1.1, 1.1])
    % print(fig,sprintf('Semi2D_3d_dFdw_x%g_y%g.pdf', x_val, y_val),'-dpdf','-r600');
    exportgraphics(fig, sprintf('Semi2D_3d_dFdw_x%g_y%g.pdf', x_val, y_val), 'ContentType', 'vector', 'Resolution', 600);

    close(fig);
end

%%
% -------------------------------
% 3. Histograms (6 total)
% -------------------------------
for i = 1:size(fixed_points,1)
    x_val = fixed_points(i,1);
    y_val = fixed_points(i,2);

    % dF/dz histogram
    files = dir(sprintf('Semi2D_histogram_dFdz_x*%g*_y*%g*_data.mat', x_val, y_val));
    if ~isempty(files)
        load(files(1).name); % dFdz
        fig = figure('Units','inches','Position',[0 0 2.3 1.6]);
        histogram(dFdz, 50, 'FaceColor', [0.85 0.33 0.10],'FaceAlpha',0.7,'Normalization','pdf', 'EdgeColor', 'none');
        xlabel('\partial F/\partial \xi'); ylabel('Density'); xlim([-1.7 1.7]);
        grid off; set(gca,'FontName','Times New Roman','GridLineStyle','--','GridAlpha',0.3);
        set(fig,'PaperUnits','inches'); set(fig,'PaperPositionMode','auto');
        box on;
        % print(fig,sprintf('Semi2D_histogram_dFdz_x%g_y%g.pdf', x_val, y_val),'-dpdf','-r600');
        exportgraphics(fig, sprintf('Semi2D_histogram_dFdz_x%g_y%g.pdf', x_val, y_val), 'ContentType', 'vector', 'Resolution', 600);
        close(fig);
    end

    % dF/dw histogram
    files = dir(sprintf('Semi2D_histogram_dFdw_x*%g*_y*%g*_data.mat', x_val, y_val));
    if ~isempty(files)
        load(files(1).name); % dFdw
        fig = figure('Units','inches','Position',[0 0 2.3 1.6]);
        histogram(dFdw, 10, 'FaceColor', [0 0.45 0.74],'FaceAlpha',0.7,'Normalization','pdf', 'EdgeColor', 'none');
        xlabel('\partial F/\partial \tau'); ylabel('Density'); xlim([-1.9 1.9]);
        grid off; set(gca,'FontName','Times New Roman','GridLineStyle','--','GridAlpha',0.3);
        set(fig,'PaperUnits','inches'); set(fig,'PaperPositionMode','auto');
        box on;
        % print(fig,sprintf('Semi2D_histogram_dFdw_x%g_y%g.pdf', x_val, y_val),'-dpdf','-r600');
        exportgraphics(fig, sprintf('Semi2D_histogram_dFdw_x%g_y%g.pdf', x_val, y_val), 'ContentType', 'vector', 'Resolution', 600);
        close(fig);
    end
end

%%

% -------------------------------
% 4. U(x,y) Surface
% -------------------------------
U_file = dir('Semi2D_U_data.mat');
if ~isempty(U_file)
    load(U_file(1).name); % X, Y, U
    fig = figure('Units','inches','Position',[0 0 3 2.6]);
    surf(X, Y, U, 'EdgeColor','none'); 
    colormap(cool); shading interp;
    camlight headlight; lighting phong;
    xlabel('x'); ylabel('y'); zlabel('U'); view(30,10);
    grid on; set(gca,'FontName','Times New Roman');
    set(fig,'PaperUnits','inches'); set(fig,'PaperPositionMode','auto');
    box on;
%     zlim([-0.5,0])
%     zlim([0, 0.5])
    zlim([-0.1, 0.1])

    % print(fig,'Semi2D_U.pdf','-dpdf','-r600');
    exportgraphics(fig, 'Semi2D_U.pdf', 'ContentType', 'vector', 'Resolution', 600);

    close(fig);
else
    warning('U(x,y) file not found.');
end

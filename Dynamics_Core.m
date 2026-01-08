function dy = Bearing_Dynamics_Out(t, y, p)
% BEARING_DYNAMICS_OUT 外圈故障动力学方程
% 输入:
%   t: 时间
%   y: 状态向量 [x_in, dx_in, y_in, dy_in, x_out, dx_out, y_out, dy_out]
%   p: 参数结构体 (由 Bearing_Config 生成)

    %% 1. 故障激励模型预计算
    % 故障几何计算
    delta_max = (p.Db/2) - sqrt((p.Db/2)^2 - (p.L_fault/2)^2);
    phi_os = 11*pi/6;         % 故障位置 (固定在外圈)
    phi_do = asin(p.L_fault/p.D_out); % 故障半波角宽度
    
    %% 2. 滚动体受力循环
    fx_sum = 0;
    fy_sum = 0;
    
    for j = 1:p.Nb
        % 当前钢球角度
        theta_j = p.wc * t + 2*pi*(j-1)/p.Nb; % 假设初始角为0
        
        % 判断是否进入外圈故障区域
        phi_pos = mod(theta_j, 2*pi); % 归一化到 0-2pi
        
        % 半正弦波故障深度函数
        h_out = 0;
        if phi_pos >= (phi_os - phi_do) && phi_pos < phi_os
            h_out = delta_max + delta_max * sin((phi_pos - phi_os)*pi/(2*phi_do));
        elseif phi_pos == phi_os
            h_out = delta_max;
        elseif phi_pos > phi_os && phi_pos <= (phi_os + phi_do)
            h_out = delta_max - delta_max * sin((phi_pos - phi_os)*pi/(2*phi_do));
        end
        
        % 动态间隙计算 (包含游隙、振动位移、故障深度)
        % 注意：y(1),y(3)是内圈位移; y(5),y(7)是外圈位移
        cd = 0.5 * p.cr * (1 - cos(abs(3*pi/2 - theta_j))); % 变间隙效应(可选)
        % 或简化为常数游隙: cd = p.cr; 
        
        delta_j = (y(1)-y(5))*cos(theta_j) + (y(3)-y(7))*sin(theta_j) - cd - h_out;
        
        % 判断是否接触 (Heaviside function)
        if delta_j > 0
            % 非线性赫兹接触力
            F_contact = p.kb * delta_j^(1.5);
            fx_sum = fx_sum + F_contact * cos(theta_j);
            fy_sum = fy_sum + F_contact * sin(theta_j);
        end
    end

    %% 3. 建立微分方程 (8自由度)
    % y = [x_i, vx_i, y_i, vy_i, x_o, vx_o, y_o, vy_o]'
    dy = zeros(8, 1);
    
    % --- 内圈方程 ---
    dy(1) = y(2);
    dy(2) = (-fx_sum + p.mi*p.e*p.w^2*cos(p.w*t) + p.Wx - p.ci*y(2) - p.ki*y(1)) / p.mi;
    
    dy(3) = y(4);
    dy(4) = (-fy_sum + p.mi*p.e*p.w^2*sin(p.w*t) - p.mi*9.81 + p.Wy - p.ci*y(4) - p.ki*y(3)) / p.mi;
    
    % --- 外圈方程 ---
    dy(5) = y(6);
    dy(6) = (fx_sum - p.co*y(6) - p.ko*y(5)) / p.mo;
    
    dy(7) = y(8);
    dy(8) = (fy_sum - p.mo*9.81 - p.co*y(8) - p.ko*y(7)) / p.mo;
    
end
function p = Bearing_Config()
% BEARING_CONFIG 集中管理轴承参数、工况及预计算
% 修改此文件中的数值以适配不同的轴承型号或工况

    %% 1. 基础几何参数 (以 6205 为例，请根据实际情况调整)
    p.D  = 52e-3;       % 外圈外径 [m]
    p.d  = 25e-3;       % 内圈内径 [m]
    p.Dm = 39.0e-3;     % 节圆直径 [m]
    p.Db = 7.94e-3;     % 滚子直径 [m]
    p.Nb = 9;           % 滚子数量
    
    % 计算导出几何参数
    p.D_out = p.Dm + p.Db; % 外圈滚道直径
    p.d_in  = p.Dm - p.Db; % 内圈滚道直径
    
    %% 2. 材料与物理参数
    p.E   = 2.07e11;    % 弹性模量 [Pa]
    p.miu = 0.3;        % 泊松比
    p.rho = 7860;       % 密度 (未使用，预留)
    
    p.mi  = 5.5;        % 内圈+轴等效质量 [kg]
    p.mo  = 12.638;     % 外圈+座等效质量 [kg]
    
    p.cr  = 5e-6;       % 径向游隙 [m]
    p.e   = 0e-3;       % 偏心距 [m] (静偏心)
    
    %% 3. 刚度与阻尼
    p.ki = 5.24e4;      % 内圈支撑刚度 [N/m]
    p.ci = 3376.84;     % 内圈支撑阻尼 [N·s/m]
    p.ko = 1.51e7;      % 外圈支撑刚度 [N/m]
    p.co = 2310.68;     % 外圈支撑阻尼 [N·s/m]
    
    %% 4. 工况设置 (用户可在此处修改转速和负载)
    p.n  = 1000;             % 转速 [rpm]
    p.w  = p.n * pi / 30;    % 角速度 [rad/s]
    p.Wx = 500;              % X方向径向载荷 [N]
    p.Wy = 0;                % Y方向径向载荷 [N]
    
    %% 5. 故障参数 (缺陷尺寸)
    p.L_fault = 2e-3;        % 缺陷宽度 [m] (默认值，具体求解脚本中可覆盖)
    
    %% 6. 预计算接触刚度 (kb) - 避免在 ODE 中重复计算
    % 沟道曲率系数
    fo = 0.525; fi = 0.515;
    Ro = fo * p.Db; Ri = fi * p.Db;
    
    % 曲率和计算
    rou_b1 = 2/p.Db; rou_b2 = 2/p.Db;
    rou_i1 = 2/p.d_in; rou_i2 = -1/Ri;
    rou_o1 = -2/p.D_out; rou_o2 = -1/Ro;
    
    rou_in  = rou_b1 + rou_b2 + rou_i1 + rou_i2;
    rou_out = rou_b1 + rou_b2 + rou_o1 + rou_o2;
    
    % 查表/拟合得到的接触变形系数
    delta_i = 0.577; delta_o = 0.68;
    
    % 赫兹接触刚度
    kbi = (2*sqrt(2)/3) * (p.E/(1-p.miu^2)) * (delta_i)^(-3/2) * (rou_in)^(-1/2);
    kbo = (2*sqrt(2)/3) * (p.E/(1-p.miu^2)) * (delta_o)^(-3/2) * (rou_out)^(-1/2);
    
    % 综合等效刚度 (写入结构体)
    p.kb = (1 / ((1/kbi)^(2/3) + (1/kbo)^(2/3)))^(3/2);
    
    %% 7. 预计算特征频率 (Hz)
    p.fr   = p.n / 60; % 转频
    p.fc   = 0.5 * p.fr * (1 - p.Db/p.Dm); % 保持架频率
    p.BPFO = p.Nb * p.fr / 2 * (1 - p.Db/p.Dm); % 外圈故障频率
    p.BPFI = p.Nb * p.fr / 2 * (1 + p.Db/p.Dm); % 内圈故障频率
    p.BPFB = p.Dm * p.fr / (2 * p.Db) * (1 - (p.Db/p.Dm)^2); % 滚动体故障频率
    
    % 保持架角速度 (rad/s)
    p.wc = 2 * pi * p.fc; 
end
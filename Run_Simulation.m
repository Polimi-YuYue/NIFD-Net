clear; clc; close all;

%% 1. 初始化与参数加载
param = Bearing_Config(); 

% --- 用户可在此处覆盖 Config 中的设置 ---
param.n = 1797;           % 修改转速
param.w = param.n*pi/30;  % 重新计算角速度
param.L_fault = 21e-3;    % 修改故障尺寸
% ------------------------------------

fprintf('当前工况: 转速 = %.2f rpm, 故障类型 = 外圈故障\n', param.n);
fprintf('理论外圈故障特征频率 (BPFO) = %.2f Hz\n', param.BPFO);

%% 2. 求解器设置
t_start = 0;
t_step  = 1e-5; % 采样间隔
t_end   = 0.5;  % 仿真总时长 (秒)
tspan   = t_start:t_step:t_end;

y0 = zeros(8, 1); % 初始状态 [位移, 速度...]
y0([1,3,5,7]) = 1e-6; % 给微小初始位移防止奇异

options = odeset('RelTol', 1e-3, 'AbsTol', 1e-6);

% 使用匿名函数传递 param 参数
[t, y] = ode45(@(t,y) Bearing_Dynamics_Out(t, y, param), tspan, y0, options);

%% 3. 数据后处理
% 去除启动时的不稳定数据 (截断前 20%)
cut_idx = round(length(t) * 0.2); 
t_steady = t(cut_idx:end);
y_steady = y(cut_idx:end, :);

% 计算加速度 (差分法)
dt = mean(diff(t_steady));
acc = diff(y_steady) / dt;
acc = [acc; acc(end, :)]; % 补齐长度

% 提取感兴趣的信号 (例如：外圈Y方向振动加速度)
signal_Ay_out = acc(:, 8); 

% 保存数据 (使用相对路径)
if ~exist('data', 'dir'), mkdir('data'); end
save('data/simulation_result.mat', 't_steady', 'y_steady', 'acc', 'param');
% writematrix(acc, 'data/acceleration_data.csv'); % 如果需要导出 Excel/CSV

%% 4. 绘图分析
figure('Color', 'w', 'Name', 'Simulation Results');

% 时域波形
subplot(2, 1, 1);
plot(t_steady, signal_Ay_out, 'b');
xlabel('Time (s)'); ylabel('Acceleration (m/s^2)');
title(['Outer Race Vibration (Y-direction), Speed: ' num2str(param.n) ' rpm']);
grid on; xlim([t_steady(1), t_steady(1)+0.2]); % 只看0.2s片段

% 包络谱分析 (调用 FFT 函数)
subplot(2, 1, 2);
signal_centered = signal_Ay_out - mean(signal_Ay_out);
envelope_signal = abs(hilbert(signal_centered));
[f, p] = FFT(t_steady, envelope_signal);

plot(f, p, 'r');
xlim([0, 500]);
xlabel('Frequency (Hz)'); ylabel('Amplitude');
title('Envelope Spectrum');
grid on;

% 标记故障特征频率
hold on;
xline(param.BPFO, 'g--', 'BPFO', 'LineWidth', 1.5);
xline(2*param.BPFO, 'g--', '2BPFO', 'LineWidth', 1.5);
xline(3*param.BPFO, 'g--', '3BPFO', 'LineWidth', 1.5);
legend('Envelope Spectrum', 'Fault Frequencies');

fprintf('仿真完成。\n');
function [f, p] = FFT(t, y)
% FFT 快速傅里叶变换工具函数
% t: 时间向量
% y: 信号向量

    tspan = t(end) - t(1);
    fs = length(t) / tspan; % 采样频率
    
    L = length(y);
    Y = fft(y);
    
    % 双边谱转单边谱
    P2 = abs(Y / L);
    P1 = P2(1 : floor(L/2)+1);
    P1(2:end-1) = 2 * P1(2:end-1);
    
    p = P1;
    f = (0 : (length(p)-1)) * (fs / L);
    
    % 确保是列向量
    f = f(:);
    p = p(:);
end
# NIFD-Net
Experimental codes for paper "A Novel Digital Twin-enabled Three-stage Feature Imputation Framework for Non-contact Intelligent Fault Diagnosis".

1. Digital Twin models
<div align=center>
<img src="https://github.com/Polimi-YuYue/NIFD-Net/blob/main/Nonlinear%20dynamic%20model%20of%20rolling%20bearing.png" width="700px">
</div>
2. Framework
<div align=center>
<img src="https://github.com/Polimi-YuYue/NIFD-Net/blob/main/Framework.png" width="500px">
</div>
3. Diagnosis Results
<div align=center>
<img src="https://github.com/Polimi-YuYue/NIFD-Net/blob/main/Diagnosis%20Results.jpg" width="500px">
</div>

# Abstract

Vibration-based fault diagnosis methods are widely used in industrial applications due to their high accuracy and reliability. However, their implementation is often hindered by challenges such as confined spaces, harsh environmental conditions, and cost limitations. Moreover, traditional deep learning-based approaches frequently overlook the critical relationship between virtual and physical signals, which can limit their diagnostic performance. To overcome these challenges, this study introduces a novel digital twin-driven feature imputation framework (NIFD-Net) for non-contact intelligent fault diagnosis. The NIFD-Net framework consists of three key stages: (1) codebook generation, where features from simulated vibration signals (SVS) and non-contact signals (NCS) are compressed and regularized using diagnosis-oriented codebooks; (2) mapping construction, where a Transformer is employed to establish the relationship between SVS and NCS in a compressed feature space; and (3) diagnostic enhancement, where imputed SVS features are integrated with original NCS features to improve diagnostic performance. First, digital twin (DT)-enabled models, developed using structural parameters and fault characteristics, dynamically simulate the operating conditions of rolling bearings to generate high-quality SVS samples. The relationship between the two feature spaces of SVS and NCS is then learned by a transformer to impute the SVS features. Finally, the imputed SVS features are integrated with the original NCS features to enhance fault diagnostic performance. Extensive experiments on three case studies demonstrate that NIFD-Net outperforms state-of-the-art methods, achieving an average accuracy of 98.34\% across diverse tasks. Notably, the framework exhibits strong noise robustness, maintaining high diagnostic performance even under adverse conditions (e.g., SNR = -10 dB). The results highlight the effectiveness of integrating simulated and real signals for non-contact fault diagnosis, offering a promising solution for industrial applications where traditional contact-based methods are impractical.


# Paper

# A Novel Digital Twin-enabled Three-stage Feature Imputation Framework for Non-contact Intelligent Fault Diagnosis

a. Yue Yu, a. Hamid Reza Karimi, b. Len Gelman, c. Xin Liu

a Department of Mechanical Engineering, Politecnico di Milano, via La Masa 1, Milan 20156, Italy

b School of Computing and Engineering, University of Huddersfield, Queensgate, Huddersfield, HD1 3DH, UK

c Department of Key Lab of Industrial Computer Control Engineering of Hebei Province, Yanshan University, Qinhuangdao 06600, China

# If this code is helpful to you, please cite this paper as follows, thank you!

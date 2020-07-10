%%% Plot One Point
% Generate a single data point, generate the plots of the fourier transform
% and spectrum in Fig 3. of JH Meng and H Riecke, 2018, Scientific Report.
% Edited by John Hongyu Meng, 06/03/2020
% Note, the Sigma= Sigma_ext /100 because of dimensionless process
%%%%
ts= 0.1; te=40; %te=40 in the paper
G2Input=0.83; Sigma=.9;%consider range 0.05 to 1
Sigma_ext=Sigma*100;
ModelParallel(G2Input,Sigma_ext,ts,te)
SynRatio=1.5;

% Copied parameters from the main code.
T_ter=te+0.1;
dt= 1e-4;
Steps=ceil(T_ter/dt);

xaxis=(1:Steps)*dt;

theta=5; V_res= -5; tau=0.020; eta=0.002; J=0.05; 
C=200; N=1000;
Ne=0; Ni =1000;  %Devided equally into two groups
Ce =0; Ci =200; 
DevideRatio = 0.7;  %How many input get within the group.
G1Input=0.8;
Cext= 800;

g=6;g0=15;       
VrevE=15;
VrevI= -VrevE;

tau_rp= 0.002;
Ratio =1.5;

% tau_l=0.00362; 
tau_l=0.002;
tau_r=0.00482; tau_d=0.00537; g_m= 0.001;
tau_c= g_m/(tau_d-tau_r);
TimePeak= tau_d*tau_r/(tau_r-tau_d)*log(tau_r/tau_d);

Mu_ext=2000;

J_ext= Sigma_ext/Mu_ext;
Nu_thr = theta/(Cext*J_ext*tau);
Nu_ext=Mu_ext^2/Sigma_ext/Cext;
Ratio=Nu_ext/Nu_thr;

sprintf("NU ext %f", Nu_ext);

%%


Mytitle=['Parallel SynRatio=',num2str(SynRatio),', G2Input=',num2str(G2Input),'Sigma=',num2str(Sigma_ext)];
FileName=['Parallel SynRatio',num2str(SynRatio),', G2Input',num2str(G2Input),'Sigma',num2str(Sigma_ext),'.mat'];
load(FileName)

%checkregion=[te/2-0.01 te/2+0.7];
checkregion=[0 te/2+0.7];
figure(50)
clf
subplot(2,1,1)
plot(xaxis,LFPG1(1:Steps))
title(['LFP Group1'])
xlim(checkregion)
xlabel('Time')
subplot(2,1,2)
plot(xaxis,LFPG2(1:Steps))
title(['LFP Group2'])
xlim(checkregion)
xlabel('Time')

figure(51)
clf
plot(LFPG1(round(Steps/2):Steps),LFPG2(round(Steps/2):Steps))
title('LFP Phase Space')
xlabel(' LFP Group 1'); ylabel(' LFP Group 2');

dataLFP=[LFPG1(round(Steps/2):Steps)',LFPG2(round(Steps/2):Steps)'];

Data1= LFPG1(round(Steps/2):Steps);
Data2= LFPG2(round(Steps/2):Steps);

Calfrac=0.75;

x1 = (0:Steps/2-1)/(te/2);
DDate1= abs(fft(Data1-mean(Data1))).^2*dt^2/(te*Calfrac);
DDate2= abs(fft(Data2-mean(Data2))).^2*dt^2/(te*Calfrac);

i_vec= round(ts/dt):round(te/dt);
xaxis=(1:Steps)/Steps*te;

PlotRegion= round(120*(te/2));

figure(52)
subplot(2,1,1)
plot(x1(1:PlotRegion),DDate1(1:PlotRegion))
ylabel('Amplitude (linear)')
title(['LFP Fourier spectrum ',Mytitle])
subplot(2,1,2)
plot(x1(1:PlotRegion),DDate2(1:PlotRegion))
ylabel('Amplitude (linear)')
hgsave(['fig52_',Mytitle,'.fig']);

% if isempty(mask1)  % Mask1 should be defined by the maximum in 
% Figure 3B7 and B8, Here just use -10 for illustation.
    mask1=-20;
% end

xaxis=x1(1:PlotRegion);
FLFP1=10*log10(DDate1(1:PlotRegion))-(mask1+30);
FLFP2=10*log10(DDate2(1:PlotRegion))-(mask1+30);

FLFP1(FLFP1<-30)=-30;
FLFP2(FLFP2<-30)=-30;
LFPD=[xaxis',FLFP1,FLFP2];
dataLFP=[LFPG1(round(Steps/2):Steps),LFPG2(round(Steps/2):Steps)];

figure(54)
subplot(2,1,1)
plot(x1(1:PlotRegion),FLFP1(1:PlotRegion))
title(['LFP Fourier spectrum ',Mytitle])
ylabel(' Amplitude (dB)')
xlim([0 60])
subplot(2,1,2)
plot(x1(1:PlotRegion),FLFP2(1:PlotRegion))
ylabel(' Amplitude (dB)')
xlim([0 60])
hgsave(['fig54_',Mytitle,'.fig']);

figure(56)
subplot(2,1,1)
semilogy(x1(1:PlotRegion),DDate1(1:PlotRegion))
xlim([0 100]); ylim([1e-3 5])
title(['LFP Fourier spectrum ',Mytitle])
subplot(2,1,2)
semilogy(x1(1:PlotRegion),DDate2(1:PlotRegion))
xlim([0 100]); ylim([1e-3 5])
hgsave(['fig56_',Mytitle,'.fig']);


function SpikeNumberVariety=ModelParallel(G2Input,Sigma_ext,ts,te)
% Generate a single data point for fig 3.
% JH Meng and H Riecke, 2018, Scientific Report.
% Edited by John Hongyu Meng, 06/03/2020
%%%%

SynRatio =1.5;
G1Input=1;
tau_l=0.002;

te=40;
Sigma_ext = 90;
G2Input = 0.84;
ts = 0.1;

theta=5; V_res= -5; tau=0.020;
 N=1000;
 Ci =200; 
DevideRatio = 0.7;  %How many input get within the group.
Cext= 800;          % How many external input sources. A dummy variable for the paper.

g0=15;       
VrevE=15;
VrevI= -VrevE;

tau_rp= 0.002;   % Refractory period, not used in the paper.

% tau_l=0.00362; 
tau_r=0.004; tau_d=0.005; g_m= 0.001;
tau_c= g_m/(tau_d-tau_r);
TimePeak= tau_d*tau_r/(tau_r-tau_d)*log(tau_r/tau_d);   % For testing

% Parameters for outputs
%ts= 0.1; te=2;   % te=40 in the paper.
T_ter=te+0.1;
dt= 1e-4;
% Steps=110000;
Steps=ceil(T_ter/dt);

% Calculate some parameter for external Poisson process, regard as one
% pocess
Mu_ext=2000;

% Dimensionless Strength
J_ext= Sigma_ext/Mu_ext;

% Theta: membrane threshold, tau membrane time constant

Nu_thr = theta/(Cext*J_ext*tau);
Nu_ext=Mu_ext^2/Sigma_ext/Cext;
Ratio=Nu_ext/Nu_thr;          % For testing.


%% Set up connection, Jmat save the connection information.

% J_ij means from J neuron to i neuron. Every neuron has Ci inhibit input.

SelfCon= Ci*DevideRatio/(N/2);
BetCon=  Ci*(1-DevideRatio)*SynRatio/(N/2);

Nd2=round(N/2);
S1mat=ones(N,N)*BetCon;
S1mat(1:Nd2,1:Nd2)=ones(Nd2,Nd2)*SelfCon;
S1mat(Nd2+1:N,Nd2+1:N)=ones(Nd2,Nd2)*SelfCon;

% V_rev = [VrevI*ones(round(Ne/2),1);VrevE*ones(round(Ni/2),1);VrevI*ones(round(Ne/2),1);VrevE*ones(round(Ni/2),1)];
V_rev = VrevI*ones(N,1);
% G_syn = [g*g0*ones(round(Ne/2),1);g0*ones(round(Ni/2),1);g*g0*ones(round(Ne/2),1);g0*ones(round(Ni/2),1)];
G_syn= g0*ones(N,1);


%% Initialize 
DelaySteps= round(tau_l/dt);
Pext= Nu_ext*dt*Cext;
                     
%Spike =cell(Steps);        % This one instore all the spikes.
V_neu= zeros(N,1);
V_neu(1:N/2)= V_res+ (theta-V_res)*rand(N/2,1);
V_neu((N/2+1):N)= V_res+ (theta-V_res)*(rand(N/2,1));

Mytitle=['Parallel',' SynRatio=',num2str(SynRatio),', G2Input=',num2str(G2Input),',Sigma=',num2str(Sigma_ext)];
FileName=['Parallel',' SynRatio',num2str(SynRatio),', G2Input',num2str(G2Input),'Sigma',num2str(Sigma_ext),'.mat'];
% Mytitle=['IsynOnly,Sigma=',num2str(Sigma_ext),', SynRatio=',num2str(SynRatio)];
% FileName=['IsynOnly,Sigma',num2str(Sigma_ext),', SynRatio=',num2str(SynRatio),'.mat'];

%% Allocate workspace
V_temp= zeros(N,1);
Ext_spike = zeros(N,1);
RefVec=zeros(N,1);
% LastSpikeTime= zeros(N,1);
NeuronG2=zeros(Steps,N/2+1);
NeuronG1=zeros(Steps,N/2+1);
LFPStd1=zeros(Steps,1);
LFPStd2=zeros(Steps,1);
InhAG2=zeros(Steps,1);
InhAG1=zeros(Steps,1);
Spike= cell(Steps,1); 

A_neu= zeros(N,1);
B_neu= zeros(N,1);
% A2_neu= zeros(N,1);
% B2_neu= zeros(N,1);
 
Phases=50;  V_sub= -10;
PhaseofCellG1= zeros(Steps,Phases);
PhaseofCellG2= zeros(Steps,Phases);
dPh= (theta- V_sub)/Phases;


%% Main loop, here I just use Euler without interpolation
% Here is a problem about how to deal with external current. Here I use a
% normal distribution. Later I'll try to use a Possion process
for i= 1:Steps
    
    V_temp = V_neu + dt* (-V_neu)/tau;
    
    % Generate external excitory synapses by Poisson process.

    %%% Only excitary neurons get input
    Ext_spike= zeros(N,1);
%     Ext_spike(1:Ne) = rand(Ne,1)<Pext ;   
%     sum(Ext_spike)/N
%     Ext_spike([1:Ne/2,N/2+1:N/2+Ne/2]) = Pext;

% % Tonic input
%   Ext_spike(1:N/2) = Pext*G1Input;
%    Ext_spike(N/2+1:N)= Pext*G2Input;
%    Mytitle=['Erev=',num2str(VrevE),', g0=',num2str(g0),', ExtMu=',num2str(Mu_ext),', SynRatio=',num2str(SynRatio),', G1Input=',num2str(G1Input),',Sigma=','tonic'];
%    FileName=['Erev',num2str(VrevE),'g0',num2str(g0),'ExtMu',num2str(Mu_ext),', SynRatio',num2str(SynRatio),', G1Input',num2str(G1Input),'Sigma','tonic','.mat'];

% Poisson input
     Ext_spike(1:N/2) = poissrnd(Pext,N/2,1)*G1Input
     Ext_spike((N/2+1):N)= poissrnd(Pext*G2Input^2,N/2,1)/G2Input;
% Periodic input
%     Ext_spike(1:Ne/2) = Pext*G1Input*2*sin(i*dt*2*pi/(0.1))^2;
%     Ext_spike((N/2+1):(N/2+Ne/2))= Pext*G2Input*2*sin(i*dt*2*pi/(0.1)-pi/2)^2;

    
    
    V_temp = V_temp+ Ext_spike* J_ext;
    
    % Influence of synapses.
%     I_inc = (V_rev-V_neu).*(A_neu-B_neu)*dt;
    V_temp= V_temp+ G_syn.*(V_rev-V_neu).*(A_neu-B_neu)*dt ;%  +(V_rev-V_neu).*(A2_neu-B2_neu)*dt;   % .. g/ tau
    
    A_neu = A_neu+dt*(-A_neu/tau_d);
    B_neu = B_neu+dt*(-B_neu/tau_r);
    
    Time= i*dt;
    if mod(i,1000)==0
        fprintf('time = %g \n',Time)
    end
     
    
    if i>DelaySteps+1 
        Spike_ind_before = Spike{i-1-DelaySteps};
        if ~isempty(Spike_ind_before)
            temp=tau_c*sum(S1mat(:,Spike_ind_before),2);
            A_neu= A_neu+ temp;
            B_neu= B_neu+ temp;
        end
    end
    
%     % Set the refractory newrons back to threshold
%     RefInd= find(RefVec>0);
%     V_temp(RefInd)=V_res;
%     RefVec= max(RefVec-1,0);

   
%     %%% For debug
%     if ShutDown> 0
%         sum(RefVec(find(V_temp>theta)))
%     end    
%     %%%%%%%%
    
    
    % Get spiking newron
    Spike_ind= find(V_temp>theta);
    Spike{i}= Spike_ind;
%     if ~isempty(Spike_ind)
%         SumRefVec=sum(RefVec);
%         RefVec(Spike_ind);
%         Spike_ind;
%         minE=min(V_neu)
%         i;
%     end
    V_temp(Spike_ind)=V_res;

    V_neu= V_temp;
   
    NeuronG1(i,1:N/2)=V_neu(1:N/2);
    NeuronG1(i,N/2+1)=mean(V_neu(1:(N/2)));
    LFPStd1(i)=std(V_neu(1:(N/2)));
    NeuronG2(i,1:N/2)=V_neu((N/2+1):N);
    NeuronG2(i,N/2+1)=mean(V_neu((N/2+1):N));
    LFPStd2(i)=std(V_neu((N/2+1):N));
  
    InhAG2(i)=mean(A_neu(N/2+1:N));
    InhAG1(i)=mean(A_neu(1:N/2));
    
    % Get phases
    
    for j=1:Phases
        
        if j==1
            PhaseCount1= sum((V_neu(  1:N/2 ) < V_sub+dPh  ) );
            PhaseCount2= sum((V_neu(  (N/2+1):N ) < V_sub+dPh  ) );
        else
            PhaseCount1= sum((V_neu(  1:N/2 ) < V_sub+dPh*j  ).*(V_neu(1:N/2 ) >= V_sub+dPh*(j-1)  )        );
            PhaseCount2= sum((V_neu(  (N/2+1):N ) < V_sub+dPh*j  ).*(V_neu((N/2+1):N ) >= V_sub+dPh*(j-1)  )        );
        end
        PhaseofCellG1(i,j)=PhaseCount1;
        PhaseofCellG2(i,j)=PhaseCount2;
    end
    
    

    
end

% Output part: let me find out!

LFPG1=NeuronG1(:,N/2+1);
LFPG2=NeuronG2(:,N/2+1);

Data1= LFPG1(round(Steps/2):Steps);
Data2= LFPG2(round(Steps/2):Steps);
Data3= InhAG1(round(Steps/2):Steps);
Data4= InhAG2(round(Steps/2):Steps);

x1 = (1:Steps/2)/(te/2);
DDate1= abs(fft(Data1-mean(Data1)));
DDate2= abs(fft(Data2-mean(Data2)));
DDate3= abs(fft(Data3-mean(Data3)));
DDate4= abs(fft(Data4-mean(Data4)));

i_vec= round(ts/dt):round(te/dt);
xaxis=(1:Steps)*dt;

PlotRegion= round(120*(te/2));



% Here is the part to calculate theta
% tstart=te/2+0.1; tend= te;
tstart=0.1; tend= te;


% tstart=3.4; tend=4;
SearchRegion= 60*(te/2);      % Interested in 0~60Hz band

[~,Indmax]= max(DDate3(1:SearchRegion));
Indmax=Indmax-1;
Fmax= x1(Indmax);
T0= 1/Fmax;

TimeSeperateVector=tstart:T0:tend;
ThetaLength= length(TimeSeperateVector);
StepVector= round(TimeSeperateVector/dt);
TimeIndexVec=zeros(ThetaLength-1,1);
TimeVec=zeros(ThetaLength-1,1);
Theta_vec=zeros(ThetaLength-1,1);
% LFPG1Sub=zeros(ThetaLength-1,1);
% LFPG2Sub=zeros(ThetaLength-1,1);

ModInd=-DelaySteps;


for iii=1:ThetaLength-1;
    [~,TimeIndexVec(iii)]=max(InhAG1(StepVector(iii):StepVector(iii+1)-1));
    TimeIndexVec(iii)=StepVector(iii)+TimeIndexVec(iii)-1;
    Theta_vec(iii)=InhAG1(TimeIndexVec(iii))-InhAG2(TimeIndexVec(iii));
%     LFPG1Sub(iii)=InhAG1(TimeIndexVec(iii));
%     LFPG2Sub(iii)=InhAG2(TimeIndexVec(iii));

end
% TimeVec=TimeIndexVec*dt;


TimeIndexVec2=TimeIndexVec;
for i=ThetaLength-2:-1:1
    if TimeIndexVec2(i+1)-TimeIndexVec2(i)<(T0/3/dt)
        TimeIndexVec2(i+1)=[];
    end
end

ThetaLength2=length(TimeIndexVec2);
% TimeVec2=TimeIndexVec2*dt;

Propotion = zeros(2,ThetaLength2-2);
for i=1:ThetaLength2-2
    for j=round(   (   TimeIndexVec2(i)+TimeIndexVec2(i+1) +2  )/2    ) :round(   (   TimeIndexVec2(i+1)+TimeIndexVec2(i+2)   )/2    )
        Propotion(1,i)=Propotion(1,i)+   sum(Spike{j}<=N/2);
        Propotion(2,i)=Propotion(2,i)+   sum(Spike{j}>N/2);
    end
end
Propotion(1,:)=Propotion(1,:)/(N/2);
Propotion(2,:)=Propotion(2,:)/(N/2);


SpikeNumberVariety=zeros(1,2);
SpikeNumberVariety(1)= var(Propotion(1,:));
SpikeNumberVariety(2)= var(Propotion(2,:));
SpikeNumberVariety


    Theta_vec2=LFPG1(TimeIndexVec2)-LFPG2(TimeIndexVec2);   %Theta
    Std_vec2=LFPStd2(TimeIndexVec2);

LFPG1=single(LFPG1);
LFPG2=single(LFPG2);


SpikeLength=round(200*T_ter);
SpikeDataMat=zeros(N,SpikeLength);
CycleIndex=1;

for i= 1:Steps
    t=i*dt;
    SpikeIndex=Spike{i};
    
    while ~isempty(find(SpikeDataMat(SpikeIndex,CycleIndex)>0, 1))
        CycleIndex=CycleIndex+1;
    end
    
    SpikeDataMat(SpikeIndex,CycleIndex)=t;
 
end

SpikeDataMat=SpikeDataMat(:,1:CycleIndex);

%% Save data
save(FileName,'Propotion','LFPG1','LFPG2','SpikeDataMat')

end

% %% Some LFPs of certain neurons
% figure(10)
% subplot(2,2,1)
% plot(xaxis,INeuron(i_vec,220),xaxis,INeuron(i_vec,221),xaxis,INeuron(i_vec,222))
% title(['Example voltages for group1, Ecells',Mytitle])
% subplot(2,2,3)
% plot(xaxis,INeuron(i_vec,420),xaxis,INeuron(i_vec,421),xaxis,INeuron(i_vec,422))
% title(['Example voltages for group1, Icells'])
% 
% subplot(2,2,2)
% plot(xaxis,ENeuron(i_vec,220),xaxis,ENeuron(i_vec,221),xaxis,ENeuron(i_vec,222))
% title(['Example voltages for group2, Ecells'])
% 
% subplot(2,2,4)
% plot(xaxis,ENeuron(i_vec,420),xaxis,ENeuron(i_vec,421),xaxis,ENeuron(i_vec,422))
% title(['Example voltages for group2, Icells'])
% 


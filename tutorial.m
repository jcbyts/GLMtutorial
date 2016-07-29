%% GLM tutorial
% This script is a tutorial for a very basic spiking Generalized Linear
% Model (GLM) similar to the one presented in Pillow et al 2008, but for a
% 1D noise stimulus instead of a spatiotemporal noise stimulus

addpath code
binSize=.001; % 1 ms bins
Duration=5; % seconds
nTimeBins=ceil(Duration/binSize);

Stim=randn(1,nTimeBins); % stimulus is gaussian white noise
% Stim=[zeros(1,100) ones(1, 800) zeros(1,100)]*1;
% Build temporal receptive field
nk=20; % number of bins to cover

% I used a difference of Gaussians to build the temporal kernel simply
% because it was easy. You could do something more sophisticated
tempRFfun=@(a,b,c,d,e) exp(-((1:nk)'-a).^2/b)-c*exp(-((1:nk)'-d).^2/e);

% baseline rate
baseline1=10*binSize; % baseline spikes per bin neuron 1
baseline2=15*binSize;  % baseline spikes per bin neuron 2

% stimulus filters
temporalRF1=5*tempRFfun(5,5,.5,8,10);
temporalRF2=5*tempRFfun(6,6,.1,10,8);

% history filters
historyFilt1=-tempRFfun(1,1,.4,2,2);
historyFilt2=-tempRFfun(1,3,.1,5,10);

% coupling filter
couple1_2=.3*tempRFfun(2,1,0,1,1);
couple2_1=.3*tempRFfun(2,2,0,1,1);

% plot this neuron
figure(1);clf
subplot(1,3,1)
plot(1:nk, temporalRF1, 1:nk, temporalRF2);
legend({'Neuron 1 RF', 'Neuron 2 RF'}, 'Location', 'Best')

subplot(1,3,2)
plot(1:nk, historyFilt1, 1:nk, historyFilt2);
legend({'Neuron 1 Hist', 'Neuron 2 Hist'}, 'Location', 'Best')

subplot(1,3,3)
plot(1:nk, couple1_2, 1:nk, couple2_1)

%% simulate GLM
kTx1=filter(temporalRF1, 1, Stim); % filter stimulus with temporal RF
kTx2=filter(temporalRF2, 1, Stim); % filter stimulus with temporal RF
% run 'help filter' to see how this implemented causal filtering
plotIt=0;


sp1=zeros(nTimeBins,1);
sp2=zeros(nTimeBins,1);

% preallocate history contribution to spike rate
hdot1=zeros(nTimeBins,1);
hdot2=zeros(nTimeBins,1);

% preallocate coupling contribution to spike rate
cdot1=zeros(nTimeBins,1);
cdot2=zeros(nTimeBins,1);

% preallocate conditional intensity (spike rate)
lambda1=zeros(nTimeBins,1);
lambda2=zeros(nTimeBins,1);

% build nonlinearity for neuron 1 and 2
g1=@(x) exp(x); % exponential
g2=@(x) 10*max(x,0); % rectified linear

figure(1); clf
subplot(6,1,1)
plot(Stim, 'k'); hold on
h=plot([t t], ylim, 'r');
% generate from a poisson process using independent Bernoulli draws for
% each bin (with probability p=rate*binSize)
t=1;
while t <= nTimeBins
    
    
    
    
    if t==1
        lambda1(t) = g1(kTx1(t) + baseline1);
        lambda2(t) = g2(kTx2(t) + baseline2);
    else
        % neuron 1 probability of spiking depends on the recent stimulus, the
        % recent spiking activity of neuron 2 and the recent history of its own
        % spiking activity
        ix=t-(1:nk);
        validIx=ix>0;
        
        ix=ix(validIx);
        
        % contribution of history
        hdot1(t)=sp1(ix)'*historyFilt1(validIx);
        hdot2(t)=sp2(ix)'*historyFilt2(validIx);
        
        
        % contribution of coupling
        cdot1(t)=sp2(ix)'*couple2_1(validIx);
        cdot2(t)=sp1(ix)'*couple1_2(validIx);
        
        
        
        
        
        
        lambda1(t) = g1(kTx1(t) + hdot1(t) + cdot1(t) + baseline1);
        lambda2(t) = g2(kTx2(t) + hdot2(t) + cdot2(t) + baseline2);
        
        if plotIt && t <= 500
            subplot(6,1,1)
            set(h, 'XData', [t t])
            
            subplot(6,1,2)
            plot(1:t, kTx1(1:t), 'b', 1:t, kTx2(1:t), 'r')
            ylim([-1 1]*max(kTx1))
            xlim([1 500])
            
            subplot(6,1,3)
            plot(1:t, hdot1(1:t), 'b', 1:t, hdot2(1:t), 'r')
            ylim([-1 1]*max(1,max(hdot1)))
            xlim([1 500])
            
            subplot(6,1,4)
            plot(1:t, cdot1(1:t), 'b', 1:t, cdot2(1:t), 'r')
            ylim([-1 1]*max(.1, max(cdot1)))
            xlim([1 500])
            
            subplot(6,1,5)
            plot(1:t, lambda1(1:t), 'b', 1:t, lambda2(1:t), 'r')
            ylim([0 1]*max(.1, 2*median(lambda1)))
            xlim([1 500])
        end
        
    end
    
    
    % independent bernoulli draw in bin t
    sp1(t)=rand<lambda1(t)*binSize;
    sp2(t)=rand<lambda2(t)*binSize;
    
    if plotIt && t <= 500
        subplot(6,1,6)
        
        if sp1(t)
            plot([t t], [0 1], 'b'); hold on
        end
        if sp2(t)
            plot([t t], [1 2], 'r');
        end
        ylim([0 2])
        xlim([1 500]);
        
        drawnow
    end
    
    t=t+1;
    
end

%%
figure(2); clf
plot(xcorr(sp1, sp2, 100));

%%
[i,j]=find([sp1 sp2]);
plot(i,j, '.')


%% build design matrix

Xstim=makeStimRows(Stim', nk);
Xspk=makeStimRows([[sp1(2:end) sp2(2:end)]; [0 0]], nk);

b=glmfit(Xstim, sp1, 'Poisson', 'link', 'log');

	




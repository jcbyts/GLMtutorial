%% GLM tutorial
% This script is a tutorial for a very basic spiking Generalized Linear
% Model (GLM) similar to the one presented in Pillow et al 2008, complete
% with spike history and coupling. For simplicity, here we will use a 1D
% noise stimulus instead of spatiotemporal noise and only two neurons, but
% the principles are the same. Additionally, this code is designed for
% clarity and is by no means optimized for speed. 
%
% For the implementation in Pillow et al 2008, 
% visit https://github.com/pillowlab/GLMspiketools
%
% jly 2016 wrote it

addpath code
rng(10294756)
% The code below generates from Poisson process where in any given bin a
% spike can occur or not occur. The probability of k events over time
% window T is given by the Poisson distribution. Remember the Poisson distribution
% is a distribution of the number of events that occur within some interval
% of time. The probability of an event occuring in each bin is a Bernoulli 
% random number. The probability of the interval between events is the
% exponential probability density function. 

% first build the parameters of the stimulus. 
binSize=.001; % 1 ms bins (Pick a size that only one or zeros events can occur)
Duration=5; % seconds
nRepeats=100; % number of repeats of the stimulus
nTimeBins=nRepeats*ceil(Duration/binSize);

Stim=repmat(smooth(randn(1,nTimeBins/nRepeats),10), nRepeats,1); % stimulus is filtered gaussian white noise

%% Build the neurons: 
% these parameters will allow us to generate from the GLM

% Build temporal receptive field
nk=20; % number of bins to cover

% Functional form for generating the different filters. This is arbitrary
% I used a difference of Gaussians to build the temporal kernel simply
% because it was easy. You could do something more sophisticated
tempRFfun=@(a,b,c,d,e) exp(-((1:nk)'-a).^2/b)-c*exp(-((1:nk)'-d).^2/e);

% Now set up the parameters for each neuron separately

% baseline rate
neuron1.baseline=10*binSize; % baseline spikes per bin neuron 1
neuron2.baseline=17*binSize;  % baseline spikes per bin neuron 2

% stimulus filters
neuron1.temporalRF=2*tempRFfun(5,5,.5,8,10);
neuron2.temporalRF=5*tempRFfun(6,6,.1,10,8);

% history filters
neuron1.historyFilt=-tempRFfun(1,1,.4,2,2);
neuron2.historyFilt=-tempRFfun(1,3,.1,5,10);

% coupling filter
neuron1.couple1=.3*tempRFfun(2,1,0,1,1);
neuron2.couple1=.3*tempRFfun(2,2,0,1,1);

% plot the neurons
figure(1);clf
subplot(1,3,1)
tx=(1:nk)*binSize;
plot(tx, neuron1.temporalRF, tx, neuron2.temporalRF);
legend({'Neuron 1', 'Neuron 2'}, 'Location', 'Best')
xlabel('Time')
title('RF')

subplot(1,3,2)
plot(tx, neuron1.historyFilt, tx, neuron2.historyFilt);
xlabel('Time')
title('History')

subplot(1,3,3)
plot(tx, neuron1.couple1, tx, neuron2.couple1)
xlabel('Time')
title('Coupling')
%% simulate GLM
% This section generates from the GLM. Because the history filters and
% coupling filters depend on what just occured in the recent time history
% of the GLM output, we have to for-loop over time and neurons to generate
% from the model. This is different than when we fit the model (next
% section).

% filter stimulus with temporal RF
kTx1=filter(neuron1.temporalRF, 1, Stim); % run 'help filter' to see how this implemented causal filtering
kTx2=filter(neuron2.temporalRF, 1, Stim);

plotIt=0; % turn this flag on to watch the GLM run for the first 500 bins

% Preallocate the requisite variables

% spikes for neuron 1 and 2
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

if plotIt
    figure(1); clf
    subplot(6,1,1)
    plot(Stim, 'k'); hold on
    h=plot([t t], ylim, 'r');
end

% loop over time
t=1;
while t <= nTimeBins
    % generate from a poisson process using independent Bernoulli draws for
    % each bin (with probability p=rate*binSize)
    
    
    
    if t==1 % there is no history yet so just use the stimulus filter and baseline
        lambda1(t) = g1(kTx1(t) + neuron1.baseline);
        lambda2(t) = g2(kTx2(t) + neuron2.baseline);
    else
        % neuron 1 probability of spiking depends on the recent stimulus, the
        % recent spiking activity of neuron 2 and the recent history of its own
        % spiking activity
        ix=t-(1:nk); % index into the recent spike history
        validIx=ix>0; % which parts of the filter are valid (only matters when t < nk)
        
        ix=ix(validIx);
        
        % contribution of history is the dot product of the history filter
        % with the recent spike history of each neuron
        hdot1(t)=sp1(ix)'*neuron1.historyFilt(validIx);
        hdot2(t)=sp2(ix)'*neuron2.historyFilt(validIx);
        
        
        % contribution of coupling is the dot product of the recent history
        % of the other neuron. With more neurons the number of coupling
        % filters scales with the number of neurons.
        cdot1(t)=sp2(ix)'*neuron1.couple1(validIx);
        cdot2(t)=sp1(ix)'*neuron2.couple1(validIx);
        
        
        % the spike rate (conditional intensity in poisson point process
        % terminology) is a nonlinear function of the sum of the stimulus,
        % history, and coupling filter outputs
        lambda1(t) = g1(kTx1(t) + hdot1(t) + cdot1(t) + neuron1.baseline);
        lambda2(t) = g2(kTx2(t) + hdot2(t) + cdot2(t) + neuron2.baseline);
        
        % plot the components of the GLM as it runs through time
        if plotIt && t <= 500
            subplot(6,1,1) % stimulus
            set(h, 'XData', [t t])
            
            subplot(6,1,2) % stimulus filtered by RF
            plot(1:t, kTx1(1:t), 'b', 1:t, kTx2(1:t), 'r')
            ylim([-1 1]*max(kTx1))
            xlim([1 500])
            
            subplot(6,1,3) % output of history filter
            plot(1:t, hdot1(1:t), 'b', 1:t, hdot2(1:t), 'r')
            ylim([-1 1]*max(1,max(hdot1)))
            xlim([1 500])
            
            subplot(6,1,4) % output of coupling filters
            plot(1:t, cdot1(1:t), 'b', 1:t, cdot2(1:t), 'r')
            ylim([-1 1]*max(.1, max(cdot1)))
            xlim([1 500])
            
            subplot(6,1,5) % conditional intensity (spike rate)
            plot(1:t, lambda1(1:t), 'b', 1:t, lambda2(1:t), 'r')
            ylim([0 1]*max(.1, 2*median(lambda1)))
            xlim([1 500])
        end
        
    end
    
    
    % independent bernoulli draw in bin t
    sp1(t)=rand<lambda1(t)*binSize;
    sp2(t)=rand<lambda2(t)*binSize;
    
    if plotIt && t <= 500
        subplot(6,1,6) % the spikes
        
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

%% plot the outut of our neurons
% plot the spike rasters and PSTHs for the stimulus repeats
% then look for coupling by plotting the cross-correlation between the two
% spike trains

% first plot the spike rasters to the repeats
raster1=reshape(sp1, nTimeBins/nRepeats, nRepeats);
raster2=reshape(sp2, nTimeBins/nRepeats, nRepeats);

% plot first 500ms of the stimulus repeats
raster1=raster1(1:500,:);
raster2=raster2(1:500,:);

[i,j]=find(raster1);
figure(1); clf
subplot(4,2,[1 3 5])
plot(i,j,'.')

ylabel('Repeats')
title('Neuron 1 raster')


[i,j]=find(raster2);
subplot(4,2,[2 4 6])
plot(i,j,'.')
ylabel('Repeats')
title('Neuron 2 raster')

subplot(4,2,7)
plot(mean(raster1,2)/binSize)
xlabel('Time (bins)')

subplot(4,2,8)
plot(mean(raster2,2)/binSize)
xlabel('Time (bins)')

figure(2); clf
[xc, lags]=xcorr(sp1, sp2, 100, 'unbiased');
plot(lags, xc, 'k');
title('Cross correlation')

%% build design matrix
% Take the time-varying stimulus and spikes and build the design matrix.
% This can be thought of as unwrapping a convolution. Each column in the
% design matrix is the temporal stimulus represented at a different lag.

% for now we're using makeStimRows to build the design matrix, but I'll
% build a more transparent version soon.
Xstim=makeStimRows(Stim, nk); % TODO: make a more transparent version of makeStimRows that can use different bases
Xspk1=makeStimRows([0; sp1(1:end-1)], nk);
Xspk2=makeStimRows([0; sp2(1:end-1)], nk);

g=@exp; % assume a simple nonlinearity for fitting

% add a bias term by augmenting the design matrix with a column of ones
X=[Xstim Xspk1 Xspk2 ones(nTimeBins,1)]; 


y=[sp1 sp2];
w0=(X'*X + eye(size(X,2)))\(X'*y); % initialize with linear regression

% minimize the negative log likelihood with a penalty on the pairwise
% difference of the weights. This penalty will impose some smoothness.
% Note: this implementation is not fast. It is meant to be intuitive to
% understand. For a fast implementation.

lambda=10; % this is a hyper-parameter. It imposes smoothness on the parameters (w).

% build the objective function as a penalized negative log-likelihood
mapfun=@(w) neglogli_poissGLM(w, X, y, g, binSize) + lambda*sum(sum(diff(w(1:nk,:)).^2)) + lambda*sum(sum(diff(w(nk+1:2*nk,:)).^2)) + lambda*sum(sum(diff(w(2*nk+1:end,:)).^2));

opts=optimset('GradObj', 'off', 'display', 'iter', 'MaxFunEvals', 100e3); % set options to display each iteration of the solver

what=fminunc(mapfun, w0, opts); % to make this faster, build a function that provides the gradient and hessian

%% plot estimated weights
rfHat=what(1:nk,:);
coupleHat=[what(2*nk+1:3*nk,1) what(nk+1:2*nk,2)];
histHat=[what(nk+1:2*nk,1) what(2*nk+1:3*nk,2)];

figure(3); clf
tx=(1:nk)*binSize;
cmap=lines;
subplot(1,3,1)
nfun=@(x) x./norm(x); % normalize for plotting
plot(tx, nfun(neuron1.temporalRF), 'Color', cmap(1,:)); hold on
plot(tx, nfun(neuron2.temporalRF), 'Color', cmap(2,:));
plot(tx, nfun(fliplr(rfHat(:,1)')), '--', 'Color', cmap(1,:))
plot(tx, nfun(fliplr(rfHat(:,2)')), '--', 'Color', cmap(2,:))
xlabel('Time')
title('RF')

subplot(1,3,2)
plot(tx, nfun(neuron1.historyFilt), 'Color', cmap(1,:)); hold on
plot(tx, nfun(neuron2.historyFilt), 'Color', cmap(2,:));
plot(tx, nfun(fliplr(histHat(:,1)')), '--', 'Color', cmap(1,:))
plot(tx, nfun(fliplr(histHat(:,2)')), '--', 'Color', cmap(2,:))
xlabel('Time')
title('History')

subplot(1,3,3)
plot(tx, nfun(neuron1.couple1), 'Color', cmap(1,:)); hold on
plot(tx, nfun(neuron2.couple1), 'Color', cmap(2,:));
plot(tx, nfun(fliplr(coupleHat(:,1)')), '--', 'Color', cmap(1,:))
plot(tx, nfun(fliplr(coupleHat(:,2)')), '--', 'Color', cmap(2,:))
xlabel('Time')
title('Coupling')

%% generate from model and plot compared to real spikes


%% Decode the stimulus with and without history/coupling



	





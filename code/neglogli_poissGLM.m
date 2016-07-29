function L = neglogli_poissGLM(w, X, y, g, dt)
% poisson negative log likelihood
% L = neglogpost_poissonGLM(w, X, y, g, dt)

L=0; % initialize at 0
for k=1:size(y,2) % loop over neurons
    L = L - poiss_logli(y(:,k), g(X*w(:,k)), dt);
end

function L = poiss_logli(r, lambda, dt)
    L=r'*log(lambda*dt) - sum(lambda);
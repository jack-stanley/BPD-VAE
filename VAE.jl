## Packages
using Flux
using Statistics
using Logging
using Test
using Random
using DataFrames
using CSV
using Plots
using Distributions
using SpecialFunctions
using StatsFuns
using Clustering

## Loading the BPD metabolomics data
df_bin = DataFrame!(CSV.File("BPD_met_runner.csv"))
x = convert(Array, df_bin)[:,5:end-8]'
t_x = x./maximum(x, dims=2)
labels = convert(Array, df_bin)[:,1:4]

## Helpers

# Helper function to return log-density of a factorized Gaussian
function factorized_gaussian_log_density(mu, logsig, x)
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((x .- mu).^2)./(σ.^2),dims=1)
end

function factorized_gaussian_log_density2(logmu, logsig, x)
  σ = exp.(logsig)
  μ = exp.(logmu)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((x .- μ).^2)./(σ.^2),dims=1)
end

# Helper function to sample from Diagonal Gaussian x~N(μ,σI)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)

# Splits the data into random batches (easier to iterate over)
function batch_data(x, batch_size=10)
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  return batch_x
end

## Model

# I'm simply using the standard Normal for my latent prior (may or may not make sense)
function log_prior(z)
  return factorized_gaussian_log_density(0, 10, z)
end

# Dh is the dimension of hidden layer, Dz is the dimension of latent space, and Ddata is obvious :)
Dh = 100
Dz = 2
Ddata = size(x)[1]

# Separates μ and logσ of z (encoder output) into an easier format to work with
function encoder_params(θ)
  μ, logσ = θ[1:Dz,:], θ[Dz+1:end,:]
  return μ, logσ
end

function mapper(θ)
  a = θ.*180
  return a
end

# The encoder function maps the data to the latent space, using parameters that we will train later
# The encoder is a multilayer perceptron with one hidden layer and a tanh non-linearity
encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, 2*Dz), encoder_params)

# The decoder function reverts the latent variables to abundance data (this is neccessary for computing logp(x|z))
# The decoder is again a multilayer perceptron with one hidden layer and a tanh non-linearity
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))

# Log-likelihood for logp(x|z)
# I chose a Normal distribution to represent the abundance values
function log_likelihood(x, z)
  a = decoder(z)
  b = -2
  # return sum(logpdf.(Truncated.(Normal.(a,b),0,180), x), dims=1)
  return factorized_gaussian_log_density(a, b, x)
end

# Computes logp(x, z)
function joint_log_density(x, z)
  return log_prior(z) + log_likelihood(x,z)
end

# Computes the likelihood of z using the trained parameters
function log_q(q_μ, q_logσ, z)
  return factorized_gaussian_log_density(q_μ, q_logσ, z)
end

# Computes the Evidence Lower Bound (ELBO)
function elbo(x)
  q_μ, q_logσ = encoder(x)
  z = sample_diag_gaussian(q_μ, q_logσ)
  joint = joint_log_density(x, z)
  q_dist = log_q(q_μ, q_logσ, z)
  elbo_estimate = mean(joint + q_dist)
  return elbo_estimate
end

# Takes the negative of ELBO; easier to minimize loss than maximize ELBO
function loss(x)
  return -elbo(x)
end

## Now to train the model
function train_model!(loss, encoder, decoder, x; nepochs = 100, lr = 1e-3)
  pars = Flux.params(decoder, encoder)
  opt = ADAM(lr)
  for i in 1:nepochs
    for d in batch_data(x)
      g_pars = Flux.gradient(pars) do
        batch_loss = loss(d)
        return batch_loss
      end
      Flux.Optimise.update!(opt, pars, g_pars)
    end
    if i%1 == 0
      @info "Loss at epoch $i: $(loss(batch_data(x)[1]))"
      @info "Value: $(x[1]); Trained: $(decoder(encoder(x)[1])[1])"
      display(scatter(encoder(x)[1][1,:], title="Latent Space Clusters with Labels", encoder(x)[1][2,:], xaxis="Latent Variable 1", yaxis="Latent Variable 2", legend=:topleft, group=labels[:,2]))
      #display(histogram(vec(decoder(encoder(t_x)[1]))))
    end
  end
  @info "Parameters of encoder and decoder trained"
end

# Training the model
train_model!(loss, encoder, decoder, x, nepochs=250, lr=1e-4)


## Graphing the output

# Getting our trained latent variables (z)
encoded_means = encoder(x)[1]


#Plotting the latent variables in the latent space
scatter(encoded_means[1,:], encoded_means[2,:], xaxis="Latent Variable 1", title="VAE Latent Space Clusters with Labels", yaxis="Latent Variable 2", legend=:topleft, group=labels[:,2])
savefig("VAE_ABD")




CSV.write("means.csv",  DataFrame(encoded_means), header=false)

Flux.params(encoder)[1]

using PyPlot
using Statistics
using Random, Distributions
using DSP: conv
using LinearAlgebra
using RHEOS
using Plots
using ProgressBars
using DelimitedFiles

include("rheos_series.jl")
include("modelfitting.jl")

function sampleMVN(mean::Vector, cov::Matrix, N::Int64=1)
    d = MvNormal(mean, cov)
    return rand(d, N)
end

function getNormalizingConstant(m::RheoModelClass, θ::NamedTuple, step_amplitude::Float64, creep::Bool)
    if m == Maxwell || m == KelvinVoigt
        return creep ? step_amplitude/θ[:k] : 
                       step_amplitude*θ[:k]
    elseif m == SLS_Zener #(:η, :kᵦ, :kᵧ)
        return creep ? step_amplitude/(θ[:kᵦ]+θ[:kᵧ]) : 
                       step_amplitude*(θ[:kᵦ]+θ[:kᵧ])
    else
        error("Unsupported model")
    end
end


function generateSeries(m::RheoModelClass, θ::Vector, cov::Matrix, rel_measure_sd::Float64, creep::Bool=true)
    # Non-Dimensional values
    N_τ = 5
    N_datapoints = 100
    N_experiments = 10

    # Calculating time duration
    τ = getTimeConstant(m, NamedTuple{m.freeparams}(θ), creep)
    t_max = N_τ * τ
    t_step = t_max/N_datapoints

    # Calculating variance for measurement noise
    step_amplitude = 10.0
    Z = getNormalizingConstant(m, NamedTuple{m.freeparams}(θ), step_amplitude, creep)
    exp_noise_var = (rel_measure_sd * Z)^2

    # Loading characteristic
    dataSeries = Vector{RheoTimeData}(undef,N_experiments)
    loading = timeline(t_start = 0, t_end = t_max, step = t_step)
    # Later add a method to dynamically identify if creep or relaxation by a loading RheoTimeData given
    loading = creep ? stressfunction(loading, hstep(offset = 0.0, amp = step_amplitude)) : 
                      strainfunction(loading, hstep(offset = 0.0, amp = step_amplitude))
    θ_samples = sampleMVN(θ, cov, N_experiments)
    for i in 1:N_experiments
        model = RheoModel(m, NamedTuple{m.freeparams}(θ_samples[:, i]))
        if creep
            dataSeries[i] = RheoTimeData(loading.σ, modelpredict(loading, model).ϵ + rand(Normal(0, exp_noise_var),length(loading.t)), loading.t, nothing)
        else
            dataSeries[i] = RheoTimeData(modelpredict(loading, model).σ + rand(Normal(0, exp_noise_var),length(loading.t)), loading.ϵ, loading.t, nothing)
        end
    end

    return dataSeries, θ_samples
end

function get_SeriesData(m::RheoModelClass, plot::Bool=false)
    if m == Maxwell || m == KelvinVoigt
        θ1 = [1.0,2.0]
        θ2 = [3.0,4.0]
        cov1 = [0.3 0 ; 0 0.5]
        cov2 = [1.0 0.5 ; 0.5 1.0]
    elseif m == SLS_Zener
        θ1 = [4.0,1.0,3.0]
        θ2 = [3.0,5.0,2.0]
        cov1 = [1. 0 0; 0 1. 0 ; 0 0 1.]
        cov2 = [1. 0.5 0.5; 0.5 1. 0.5 ; 0.5 0.5 1.]
    end
    rel_measure_sd = 0.3
    series1, θ_series1 = generateSeries(m, θ1, cov1, rel_measure_sd)
    series2, θ_series2 = generateSeries(m, θ2, cov2, rel_measure_sd)
    if plot plot_θ_samples(θ1, θ2, cov1, cov2, hcat(θ_series1, θ_series2)) end
    return RheoExperimentSeries([series1 ; series2]), hcat(θ_series1, θ_series2)
end

function plot_θ_samples(θ1::Vector, θ2::Vector, cov1::Matrix, cov2::Matrix, θ_samples::Matrix)

    d1 = MvNormal(θ1, cov1)
    d2 = MvNormal(θ2, cov2)

    z1 = θ_samples[:,1:10]
    z2 = θ_samples[:,11:20]

    x = range(0,10,100)
    y = range(0,10,100)
    Z1 = [pdf(d1,[i,j]) for i in x, j in y]
    Z2 = [pdf(d2,[i,j]) for i in x, j in y]


    fig, ax = subplots(figsize=(8,6))
    ax.scatter(z1[2,:],z1[1,:],marker = "x", s=100, color = "green")
    ax.scatter(z2[2,:],z2[1,:],marker = "x", s=100, color = "red")
    ax.contour(x, y, Z1, cmap="viridis")
    ax.contour(x, y, Z2, cmap="magma")
    ax.set_xlabel("k", loc="right", fontsize = 28)
    ax.set_ylabel("η", loc="top", rotation=0, fontsize = 28)
    PyPlot.savefig(joinpath("figures","two_series","theta_samples.png"))
    PyPlot.close()
end

function displayTestData(dataSeries::RheoExperimentSeries)

    fig, ax = subplots(figsize=(8,6))
    for i in eachindex(dataSeries.data)
        ax.plot(dataSeries.data[i].t, dataSeries.data[i].ϵ,
                color = i<=10 ? "green" : "blue",
                label = i<=10 ? "Set 1" : "Set 2")
    end
    ax.set_xlabel("Time", fontsize = 28)
    ax.set_ylabel("Strain", fontsize = 28)
    ax.legend()
    gcf()
    #PyPlot.savefig(joinpath("figures","two_series","sls_raw.png"))
end

using RHEOS
using DSP: conv

using LinearAlgebra
using Plots
using ProgressBars

using StatsBase
using MultivariateStats, Statistics, StatsAPI
using Clustering

include("rheos_series.jl")

"""
    get_feature_matrix(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)

convert a feature dictionary with selected keys in `feature_list` into a matrix.
Each column represents an observation.

Order of rows depends on dictionary internal ordering, not 'feature_list' parameter.
"""
function get_feature_matrix(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    if isempty(dataSeries.features) error("No features to analyse") end
    if isnothing(feature_list)
        feature_list = collect(keys(dataSeries.features))
        filter!(l->l≠:label,feature_list)
    else
        for feature in feature_list
            if feature ∉ collect(keys(dataSeries.features)) error("$feature does not exist yet") end
        end
    end

    return Matrix{Float64}(reduce(hcat,[dataSeries.features[feature] for feature in feature_list])') 
end


"""
    standardize_matrix(x::Matrix{Float64})

standardize features in matrix `x` by subtracting mean and dividing by standard deviation.
"""
function standardize_matrix(x::Matrix{Float64})
    x_bar = Matrix{Float64}(undef, size(x))
    for i in 1:size(x)[1]
        x_bar[i,:] = (x[i,:] .- mean(x[i,:])) ./ (var(x[i,:])^0.5)
    end
    return x_bar
end


"""
do_PCA(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing, maxoutdim::Union{Nothing, Int64} = nothing)

does PCA as provided in MultivariateStats.

# Arguments
    
    - `dataSeries`: `RheoExperimentSeries` struct containing all data and features
    - `feature_list`: `Vector` optional parameter selects which features to use for PCA (default is all features)
    - `maxoutdim`: `Int64` optional parameter determining maximum output dimension of PCA (default is the number of features)
"""
function do_PCA(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing, maxoutdim::Union{Nothing, Int64} = nothing)
    features = get_feature_matrix(dataSeries, feature_list)
    standardized_features = standardize_matrix(features)

    maxoutdim = isnothing(maxoutdim) ? size(features)[1] : maxoutdim

    return MultivariateStats.fit(PCA, standardized_features, maxoutdim=maxoutdim)
end


"""
    do_LDA(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing, maxoutdim::Union{Nothing, Int64} = nothing)

does LDA as provided in MultivariateStats.
Compatible with any number of labels.

# Arguments
    
    - `dataSeries`: `RheoExperimentSeries` struct containing all data and features
    - `feature_list`: `Vector` optional parameter selects which features to use for PCA (default is all features)
    - `maxoutdim`: `Int64` optional parameter determining maximum output dimension of PCA (default is the number of features)
"""
function do_LDA(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    features = get_feature_matrix(dataSeries, feature_list)
    return StatsAPI.fit(MulticlassLDA, features, dataSeries.features[:label], outdim=2)
end


"""
    do_LDA(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)

does 2-Class LDA as provided in MultivariateStats.

# Arguments
    
    - `dataSeries1`: `RheoExperimentSeries` struct containing all data and features for the 1st label
    - `dataSeries2`: `RheoExperimentSeries` struct containing all data and features for the 2nd label
    - `feature_list`: `Vector` optional parameter selects which features to use for PCA (default is all features)
"""
function do_LDA(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    features1 = get_feature_matrix(dataSeries1, feature_list)
    features2 = get_feature_matrix(dataSeries2, feature_list)
    return StatsAPI.fit(LinearDiscriminant, features1, features2)
end


"""
    do_Clustering(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)

does kmeans Clustering on select features.

# Arguments
    
    - `dataSeries`: `RheoExperimentSeries` struct containing all data and features
    - `feature_list`: `Vector` optional parameter selects which features to use for PCA (default is all features)
"""
function do_Clustering(dataSeries::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    features = get_feature_matrix(dataSeries, feature_list)
    return Clustering.kmeans(features, length(Set(dataSeries.features[:label])))
end


"""
    KLdivergence_mvGaussian(μ₁::Matrix{Float64}, Σ₁::Matrix{Float64}, μ₂::Matrix{Float64}, Σ₂::Matrix{Float64})

helper function for KLDivergence().
"""
function KLdivergence_mvGaussian(μ₁::Matrix{Float64}, Σ₁::Matrix{Float64}, μ₂::Matrix{Float64}, Σ₂::Matrix{Float64})
    Σ₂_inv = inv(Σ₂)
    return 0.5 * (log(det(Σ₂)/det(Σ₁)) .- length(μ₁) .+ tr(Σ₂_inv*Σ₁) .+ (μ₂-μ₁)' * Σ₂_inv * (μ₂-μ₁))
end


"""
    KLDivergence(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)

calculate and return KLDivergence between two dataSeries based on select features that are assumed to be samples from a multivariate gaussian.
"""
function KLDivergence(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    features1 = get_feature_matrix(dataSeries1, feature_list)
    mean1 = mean(features1, dims=2)
    cov1 = Statistics.cov(features1, dims=2)
    features2 = get_feature_matrix(dataSeries2, feature_list)
    mean2 = mean(features2, dims=2)
    cov2 = Statistics.cov(features2, dims=2)
    return KLdivergence_mvGaussian(mean1, cov1, mean2, cov2)
end


"""
    symmetric_KLDivergence(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)

calculate and return symmetric KLDivergence between two dataSeries based on select features that are assumed to be samples from a multivariate gaussian.
"""
function symmetric_KLDivergence(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    features1 = get_feature_matrix(dataSeries1, feature_list)
    mean1 = mean(features1, dims=2)
    cov1 = Statistics.cov(features1, dims=2)
    features2 = get_feature_matrix(dataSeries2, feature_list)
    mean2 = mean(features2, dims=2)
    cov2 = Statistics.cov(features2, dims=2)
    return KLdivergence_mvGaussian(mean1, cov1, mean2, cov2) + KLdivergence_mvGaussian(mean2, cov2, mean1, cov1)
end


"""
    HellingerDistance(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)

calculate and return Hellinger Distance between two dataSeries based on select features that are assumed to be samples from a multivariate gaussian.
"""
function HellingerDistance(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, feature_list::Union{Vector{Symbol}, Nothing} = nothing)
    features1 = get_feature_matrix(dataSeries1, feature_list)
    μ₁ = mean(features1, dims=2)
    Σ₁ = Statistics.cov(features1, dims=2)
    features2 = get_feature_matrix(dataSeries2, feature_list)
    μ₂ = mean(features2, dims=2)
    Σ₂ = Statistics.cov(features2, dims=2)
    return 1 .-(det(Σ₁)^(1/4)*det(Σ₁)^(1/4))/(det(0.5*Σ₁+0.5*Σ₂)^(1/2))*exp(-1/8* (μ₂-μ₁)' * (inv(0.5*Σ₁+0.5*Σ₂) * (μ₂-μ₁)))
end


"""
    HellingerDistance(x1::Matrix, x2::Matrix)

alternate function for HellingerDistance() taking feature matrices as inputs directly.
"""
function HellingerDistance(x1::Matrix, x2::Matrix)
    μ₁ = mean(x1, dims=2)
    Σ₁ = Statistics.cov(x1, dims=2)
    μ₂ = mean(x2, dims=2)
    Σ₂ = Statistics.cov(x2, dims=2)
    return 1 .-(det(Σ₁)^(1/4)*det(Σ₁)^(1/4))/(det(0.5*Σ₁+0.5*Σ₂)^(1/2))*exp(-1/8* (μ₂-μ₁)' * (inv(0.5*Σ₁+0.5*Σ₂) * (μ₂-μ₁)))
end


"""
    KLDivergence(x1::Matrix, x2::Matrix)

alternate function for KLDivergence() taking feature matrices as inputs directly.
"""
function KLDivergence(x1::Matrix, x2::Matrix)
    mean1 = mean(x1, dims=2)
    cov1 = Statistics.cov(x1, dims=2)
    mean2 = mean(x2, dims=2)
    cov2 = Statistics.cov(x2, dims=2)
    return KLdivergence_mvGaussian(mean1, cov1, mean2, cov2)
end


"""
    get_residuals(dataSeries::RheoExperimentSeries, m::RheoModelClass, modloading::LoadingType=strain_imposed)

calculate and return average residuals over time leftover from fitting model `m` to each experiment's data in dataSeries.
Assumes modelfit!() has already been done on dataSeries.
"""
function get_residuals(dataSeries::RheoExperimentSeries, m::RheoModelClass, modloading::LoadingType=strain_imposed)
    residuals = zeros(size(dataSeries.data[1].t))
    for i in eachindex(dataSeries.data)
        model = RheoModel(m, NamedTuple{m.freeparams}([dataSeries.features[para][i] for para in m.freeparams]))
        if modloading==strain_imposed
            data_ext = extract(dataSeries.data[i], strain_only)
            prediction = modelpredict(data_ext, model)
            residuals = residuals .+ (dataSeries.data[i].σ .- prediction.σ) 
        else
            data_ext = extract(dataSeries.data[i], strain_only)
            prediction = modelpredict(data_ext, model)
            residuals = residuals .+ (dataSeries.data[i].ϵ .- prediction.ϵ)
        end
    end
    return residuals
end


"""
    get_residuals(dataSeries::RheoTimeData, m::RheoModel, modloading::LoadingType=strain_imposed)

calculate and return residuals over time leftover from fitting model `m` to the data in dataSeries.
"""
function get_residuals(dataSeries::RheoTimeData, m::RheoModel, modloading::LoadingType=strain_imposed)
    residuals = zeros(size(dataSeries.t))
    if modloading==strain_imposed
        data_ext = extract(dataSeries, strain_only)
        prediction = modelpredict(data_ext, m)
        residuals = residuals .+ (dataSeries.σ .- prediction.σ) 
    else
        data_ext = extract(dataSeries, strain_only)
        prediction = modelpredict(data_ext, m)
        residuals = residuals .+ (dataSeries.ϵ .- prediction.ϵ)
    end
    return residuals
end


"""
    AIC_residual(residuals::Vector{Float64}, k::Int64)

calculate and return the Akaike Information Criterion based on the residual vector assuming residuals are normally distributed.
"""
function AIC_residual(residuals::Vector{Float64}, k::Int64)
    n = length(residuals)
    return 2*k - n*log(sum(residuals.^2)/n)
end

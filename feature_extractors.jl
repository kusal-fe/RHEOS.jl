using RHEOS
using DSP: conv
using Base
using LinearRegression

include("rheos_series.jl")


"""
    modelfit!(dataSeries::RheoExperimentSeries, m::RheoModelClass, θ_initial::Union{Nothing, Vector{Float64}}=nothing, modloading::LoadingType=stress_imposed;)

    call modelfit on each RheoTimeData in DataSeries.data and saving model parameters in vectors with corresponding keys in DataSeries.features for each experiment.
    E.g. (DataSeries.features["η" : [..., 1.5, ...])

    Index i of experiment in DataSeries.data corresponds to its feature being in the ith index of that feature vector.
    
    # Arguments
    
    - `dataSeries`: `RheoExperimentSeries` struct containing all data and features
    - `model`: `RheoModelClass` containing moduli functions and named tuple parameters
    - `modloading`: `strain_imposed` or `1`, `stress_imposed` or `2`
    - `θ_initial`: Initial parameters to use in fit (uses 0.5 for all parameters if not defined), provided as a Vector if used for all experiment, or as a matrix with each row used for each experiment. (near equivalent to p0 in modelfit(RheoTimeData))
    - `θ_tol`: Optional Parameter that calculates upper and lower bounds for parameters, provided as a Vector (calculates l0 and h0 for modelfit(RheoTimeData))
"""
function modelfit!(dataSeries::RheoExperimentSeries, m::RheoModelClass, θ_initial::Union{Nothing, Vector{Float64}}=nothing, modloading::LoadingType=stress_imposed;)
    
    N = length(dataSeries.data)
    θ_initial = isnothing(θ_initial) ? NamedTuple{m.freeparams}([0.5 for p in 1:length(m.freeparams)]) : NamedTuple{m.freeparams}(θ_initial)
    for key in m.freeparams
        dataSeries.features[key] = Vector{}(undef, N)
    end

    for i in 1:N
        model = modelfit(dataSeries.data[i], m, modloading, p0=θ_initial)
        θ = getparams(model)
        for key in keys(θ)
            dataSeries.features[key][i] = θ[key]
        end
    end
end

function modelfit!( dataSeries::RheoExperimentSeries,
                    m::RheoModelClass,
                    θ_initial::Vector{Float64},
                    θ_tol::Vector{Float64},
                    modloading::LoadingType=stress_imposed;)
    
    N = length(dataSeries.data)
    p0 = NamedTuple{m.freeparams}(θ_initial)
    lo = NamedTuple{m.freeparams}(max.(θ_initial-θ_tol, zeros(size(θ_initial))))
    hi = NamedTuple{m.freeparams}(θ_initial+θ_tol)
    
    for key in m.freeparams
        dataSeries.features[key] = Vector{}(undef, N)
    end

    for i in 1:N
        model = modelfit(dataSeries.data[i], m, modloading, p0=p0,lo=lo,hi=hi)
        θ = getparams(model)
        for key in keys(θ)
            dataSeries.features[key][i] = θ[key]
        end
    end
end

function modelfit!(dataSeries::RheoExperimentSeries, m::RheoModelClass, θ_initial::Matrix{Float64}, modloading::LoadingType=stress_imposed)
    
    N = length(dataSeries.data)
    for key in m.freeparams
        dataSeries.features[key] = Vector{}(undef, N)
    end

    for i in 1:N
        print(size(θ_initial))
        model = modelfit(dataSeries.data[i], m, modloading,
                         p0=NamedTuple{m.freeparams}(θ_initial[:,i]))
        θ = getparams(model)
        for key in keys(θ)
            dataSeries.features[key][i] = θ[key]
        end
    end
end


"""
    get_gradient!(dataSeries::RheoExperimentSeries, time_range::Tuple{Float64,Float64}, modloading::LoadingType=stress_imposed)

    fit a linear regression model to stress or strain (depending on modloading) within a time range and saving the slope and intercept as features for each experiment.

    # Arguments
    
    - `dataSeries`: `RheoExperimentSeries` struct containing all data and features
    - `time_range`: `Tuple{Float64,Float64}` containing start and end times inclusive (times must exist as datapoints)
    - `modloading`: `strain_imposed` or `1`, `stress_imposed` or `2`
"""
function get_gradient!(dataSeries::RheoExperimentSeries, time_range::Tuple{Float64,Float64}, modloading::LoadingType=stress_imposed)
    dataSeries.features[:grad] = Vector{}(undef, length(dataSeries.data))
    dataSeries.features[:intercept] = Vector{}(undef, length(dataSeries.data))
    for (i, data) in enumerate(dataSeries.data)
        t1 = findfirst(x -> x == time_range[1], data.t)
        t2 = findfirst(x -> x == time_range[2], data.t)
        lr = modloading==stress_imposed ? linregress(data.t[t1:t2], data.ϵ[t1:t2]) : linregress(data.t[t1:t2], data.σ[t1:t2])
        dataSeries.features[:grad][i] = LinearRegression.slope(lr)[1]
        dataSeries.features[:intercept][i] = LinearRegression.bias(lr)
    end
end


"""
    add_τ!(dataSeries::RheoExperimentSeries, m::RheoModelClass, creep::Bool)

    call get_τ to determine time constant based on model m (assumes modelfit!() done already) for each experiment.

    Not used - Needs to be removed or updated
"""
function add_τ!(dataSeries::RheoExperimentSeries, m::RheoModelClass, creep::Bool)
    τ = get_τ(m, creep)
    dataSeries.features[:τ] = τ(dataSeries.features)
end


"""
    log10!(dataSeries::RheoExperimentSeries, feature::Symbol)

    add a new feature that is simply the base 10 log of select feature for each experiment.
"""
function log10!(dataSeries::RheoExperimentSeries, feature::Symbol)
    dataSeries.features[Symbol(string("log_"),feature)] = log10.(dataSeries.features[feature])
end


"""
    multiply!(dataSeries::RheoExperimentSeries, feature1::Symbol, feature2::Symbol, feature_name::Union{Nothing,Symbol} = nothing)

    add a new feature that is simply the product of two select features for each experiment.

    Custom key name for the feature can be provided under `feature_name`
"""
function multiply!(dataSeries::RheoExperimentSeries, feature1::Symbol, feature2::Symbol, feature_name::Union{Nothing,Symbol} = nothing)
    feature_name = isnothing(feature_name) ? Symbol(feature1,feature2) : feature_name
    if feature_name ∈ keys(dataSeries.features)
        error("$feature_name already exists")
    end
    dataSeries.features[feature_name] = dataSeries.features[feature1].*dataSeries.features[feature2]
end


"""
    divide!(dataSeries::RheoExperimentSeries, feature1::Symbol, feature2::Symbol, feature_name::Union{Nothing,Symbol} = nothing)
    
    add a new feature that is feature1/feature2 for each experiment.

    Custom key name for the feature can be provided under `feature_name`
"""
function divide!(dataSeries::RheoExperimentSeries, feature1::Symbol, feature2::Symbol, feature_name::Union{Nothing,Symbol} = nothing)
    feature_name = isnothing(feature_name) ? Symbol(String(feature1,"_div_",feature2)) : feature_name
    if feature_name ∈ keys(dataSeries.features)
        error("$feature_name already exists")
    end
    dataSeries.features[feature_name] = dataSeries.features[feature1].*dataSeries.features[feature2]
end


"""
    include!(dataSeries::RheoExperimentSeries, feature_vector::Vector{Any}, feature_name::Symbol)
    
    save a manual `feature_vector`` under the key `feature_name` in dataSeries.features
"""
function include!(dataSeries::RheoExperimentSeries, feature_vector::Vector{Any}, feature_name::Symbol)
    if length(feature_vector) != length(dataSeries.data)
        error("Feature list size does not match number of series")
    end
    dataSeries.features[feature_name] = feature_vector
end


"""
    custom_feature!(dataSeries::RheoExperimentSeries, extractor::Function, feature_name::Symbol)
    
    add a new feature that is a custom function of already present features based on `extractor` and saved under key `feature_name`

    Extractor must take a Dict as an input (To be more generalized later)
"""
function custom_feature!(dataSeries::RheoExperimentSeries, extractor::Function, feature_name::Symbol)
    dataSeries.features[feature_name] = extractor(dataSeries.features)
end


"""
    custom_datafeature!(dataSeries::RheoExperimentSeries, extractor::Function, feature_name::Symbol)
    
    add a new feature that is a custom function of stress-strain-time data based on `extractor` and saved under key `feature_name`

    Extractor must take a RheoTimeData as an input (To be more generalized later)
"""
function custom_datafeature!(dataSeries::RheoExperimentSeries, extractor::Function, feature_name::Symbol)
    dataSeries.features[feature_name] = extractor.(dataSeries.data)
end


"""
    delete!(dataSeries::RheoExperimentSeries, feature_name::Symbol)
    
    delete the feature vector under the key `feature_name`
"""
function delete!(dataSeries::RheoExperimentSeries, feature_name::Symbol)
    Base.delete!(dataSeries.features, feature_name)
end
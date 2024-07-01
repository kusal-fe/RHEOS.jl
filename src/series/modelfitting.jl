"""
Given a set of time series data and average the strain data and returns it
"""
function avgCurves(dataSeries::Vector{RheoTimeData}, creep::Bool=true)
    loaded_data=Array{Vector{Float64}}(undef,length(dataSeries))
    for i in eachindex(dataSeries)
        loaded_data[i] = creep ? dataSeries[i].ϵ : dataSeries[i].σ
    end
    predictions = reduce(hcat,loaded_data)
    if creep
        return RheoTimeData(dataSeries[eachindex(dataSeries)[1]].σ, mean(predictions, dims=2)[:,1], dataSeries[eachindex(dataSeries)[1]].t, nothing)
    else
        return RheoTimeData(mean(predictions, dims=2)[:,1], dataSeries[eachindex(dataSeries)[1]].ϵ, dataSeries[eachindex(dataSeries)[1]].t, nothing)
    end
end

"""
Given a set of series of RheoTimeData, and a desired model,
the function fits a model to the time average of RheoTimeData
"""
function avgModelFit(dataSeries::Vector{RheoTimeData}, m::RheoModelClass, creep::Bool=true)
    data = avgCurves(dataSeries)
    model = modelfit(data, m, stress_imposed, p0=NamedTuple{m.freeparams}([0.5 for p in 1:length(m.freeparams)]))
    return model, data
end
function avgModelFit(dataSeries::Vector{RheoTimeData}, m::RheoModelClass, τ_cutoff::Float64, creep::Bool=true)
    return avgModelFit(dataSeries, m, creep)
end

"""
Given a set of series of RheoTimeData, and a desired model,
the function fits a model to each RheoTimeData and averages their parameters to give final model parameters 
"""
function avgParametersFit(dataSeries::Vector{RheoTimeData}, m::RheoModelClass)
    N = length(dataSeries)
    parameters=Vector{Vector{Float64}}(undef,N)
    models = Vector{RheoModel}(undef,N)
    for i in 1:length(dataSeries)
        model = modelfit(dataSeries[i], m, stress_imposed, p0=NamedTuple{m.freeparams}([0.5 for p in 1:length(m.freeparams)]))
        l =getparams(model)
        parameters[i] = [l[i] for i in keys(l)]
        models[i] = model
    end
    parameters = reduce(hcat,parameters)
    avg_parameters = mean(parameters, dims=2)[:,1]
    model = RheoModel(m, NamedTuple{m.freeparams}(avg_parameters))
    return model, models
end

function getTimeConstant(m::RheoModelClass, θ::NamedTuple, creep::Bool=true)
    time_constant_default = 5.0
    if m == Maxwell
        return creep ? time_constant_default : 
                       θ[:η]/θ[:k]
    elseif m == KelvinVoigt
        return creep ? θ[:η]/θ[:k] : 
                       time_constant_default
    elseif m == SLS_Zener
        return creep ? θ[:η]*(θ[:kᵦ]+θ[:kᵧ])/(θ[:kᵦ]*θ[:kᵧ]) : 
                       θ[:η]/θ[:kᵦ]
    else
        error("Unsupported Model")
    end
end

function avgParametersFit(dataSeries::Vector{RheoTimeData}, m::RheoModelClass, τ_cutoff::Float64, creep::Bool=true)
    #θ_initial = NamedTuple{m.freeparams}([1,2,3])
    #model_0,_ = avgModelFit(dataSeries, m, creep)
    if m ∈ (Maxwell, KelvinVoigt) 
        θ_default = [4.0,7.0] 
        θ_initial = NamedTuple{m.freeparams}(θ_default)
    else
        if m == SLS_Zener 
            θ_default = [1.0,2.0,3.0] 
        else 
            θ_default = [1.0 for i in eachindex(m.freeparams)]
        end
        #θ_initial = getTimeConstant(m, getparams(model_0), creep) ≤ τ_cutoff ? getparams(model_0) : NamedTuple{m.freeparams}(θ_default)
        θ_initial = NamedTuple{m.freeparams}(θ_default)
    end
    N = length(dataSeries)
    parameters=Vector{}(undef,N)
    models = Vector{RheoModel}(undef,N)
    for i in 1:length(dataSeries)
        model = modelfit(dataSeries[i], m, stress_imposed, p0=θ_initial)
        l = getparams(model)
        parameters[i] = getTimeConstant(m, l, creep) ≤ τ_cutoff ? [l[i] for i in keys(l)] : missing
        models[i] = model
    end
    if isempty(skipmissing(parameters))
        println("All Parameters are poor fitting")
        parameters = reduce(hcat,parameters)
    else
        parameters = reduce(hcat,skipmissing(parameters))
    end
    avg_parameters = mean(parameters, dims=2)[:,1]
    model = RheoModel(m, NamedTuple{m.freeparams}(avg_parameters))
    return model, models
end

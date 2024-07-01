using RHEOS
using DSP: conv


"""
    RheoExperimentSeries(data::Vector{RheoTimeData}}

`RheoExperimentSeries` mutable struct contains an array of RheoTimeData, each representing
stress-strain time data for a single experiment.
The struct also contains a features dictionary representing relevant faetures for analysis
(e.g. fitted model parameters) 

If preferred, entries in features dictionary can be input manually at initialisation.

# Fields

- `data`: Array of RheoTimeData
- `features`: Dictionary of Features Extracted (Initialized Empty)
"""
@kwdef mutable struct RheoExperimentSeries
    data::Vector{RheoTimeData}
    features::Dict{Symbol, Vector{Any}} = Dict()
end

function RheoExperimentSeries(data::Vector{RheoTimeData})

    #RheoExperimentSeries(data, Dict(),
    #               isnothing(log) ? loginit(savelog, :RheoExperimentSeries, params = NamedTuple(), keywords = (data=data,comment=comment),
    #                                        info=(comment=comment, type=typecheck) )
    #                              : log )
    RheoExperimentSeries(data, Dict())
end

Base.copy(x::RheoExperimentSeries) = RheoExperimentSeries(x.data, x.features)


"""
    get_τ(m::RheoModelClass, creep::Bool)

return a τ(θ) function for Maxwell, Kelvin-Voigt and SLS_Zener models.

Unused - method should get updated later or removed.
"""
function get_τ(m::RheoModelClass, creep::Bool)
    if m == Maxwell
        if creep
            error("No τ exists")
        else
            return(θ -> θ[:η]./θ[:k])
        end
    elseif m == KelvinVoigt
        if creep
            return(θ ->θ[:η]./θ[:k])
        else
            error("No τ exists")
        end
    elseif m == SLS_Zener
        if creep
            return (θ ->θ[:η].*(θ[:kᵦ].+θ[:kᵧ])./(θ[:kᵦ].*θ[:kᵧ])
)        else
            return(θ -> τ(θ) = θ[:η]./θ[:kᵦ])
        end
    else
        error("Unsupported Model. Please custom define τ")
    end
end


"""
    label!(dataSeries::RheoExperimentSeries, label::String)

modifies dataSeries by adding label as a feature to all entries in dataSeries.
"""
function label!(dataSeries::RheoExperimentSeries, label::String)
    dataSeries.features[:label] = fill(label, length(dataSeries.data))
end


"""
    combine!(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries)

append dataSeries2.data to dataSeries1.data and dataSeries2.features to dataSeries1.features.
Features in dataSeries1 and dataSeries2 must match.
"""
function combine!(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries)
    if keys(dataSeries1.features) != keys(dataSeries2.features)
        error("Features in the two models do not match")
    end
    if :label ∉ keys(dataSeries1.features) || :label ∉ keys(dataSeries2.features)
        label!(dataSeries1, "1")
        label!(dataSeries2, "2")
    end
    dataSeries1.data = [dataSeries1.data ; dataSeries2.data]
    for key in keys(dataSeries1.features)
        dataSeries1.features[key] = [dataSeries1.features[key] ; dataSeries2.features[key]]
    end
end


"""
    combine(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries)

return a RheoExperimentSeries with concatenated data and features from dataSeries1 and dataSeries2.
Features in dataSeries1 and dataSeries2 must match.
"""
function combine(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries)
    if keys(dataSeries1.features) != keys(dataSeries2.features)
        error("Features in the two models do not match")
    end
    if :label ∉ keys(dataSeries1.features) || :label ∉ keys(dataSeries2.features)
        label!(dataSeries1, "1")
        label!(dataSeries2, "2")
    end
    features = Dict()
    for key in keys(dataSeries1.features)
        features[key] = [dataSeries1.features[key] ; dataSeries2.features[key]]
    end
    return RheoExperimentSeries([dataSeries1.data ; dataSeries2.data], features)
end


"""
    combine!(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries)

append dataSeries2.data to dataSeries1.data and dataSeries2.features to dataSeries1.features.
Features in dataSeries1 and dataSeries2 must match.

added label1 and label2 combines label! functionality to dataSeries1 and dataSeries2 respectively as well.
"""
function combine!(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, label1::String, label2::String)
    if keys(dataSeries1.features) != keys(dataSeries2.features)
        error("Features in the two models do not match")
    end
    label!(dataSeries1, label1)
    label!(dataSeries2, label2)
    dataSeries1.data = [dataSeries1.data ; dataSeries2.data]
    for key in keys(dataSeries1.features)
        dataSeries1.features[key] = [dataSeries1.features[key] ; dataSeries2.features[key]]
    end
end


"""
    combine(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, label1::String, label2::String)

return a RheoExperimentSeries with concatenated data and features from dataSeries1 and dataSeries2.
Features in dataSeries1 and dataSeries2 must match.

added label1 and label2 combines label! functionality to dataSeries1 and dataSeries2 respectively as well.
"""
function combine(dataSeries1::RheoExperimentSeries, dataSeries2::RheoExperimentSeries, label1::String, label2::String)
    if keys(dataSeries1.features) != keys(dataSeries2.features)
        error("Features in the two models do not match")
    end
    features = Dict()
    for key in keys(dataSeries1.features)
        features[key] = [dataSeries1.features[key] ; dataSeries2.features[key]]
    end
    features[:label] = [fill(label1, length(dataSeries1.data)) ; fill(label2, length(dataSeries2.data))]
    return RheoExperimentSeries([dataSeries1.data ; dataSeries2.data], features)
end


"""
    matrix_to_features(features::Matrix, keys::Vector{Symbol})

convert a feature matrix `features` back to dictionary given `keys` and return dict
"""
function matrix_to_features(features::Matrix, keys::Vector{Symbol})
    if size(features)[0] != length(keys) error("Number of features does not match number of keys given") end
    feature_dict = Dict()
    for i in eachindex(keys)
        feature_dict[keys[i]] = features[i,:]
    end
    return feature_dict
end


"""
    mask_out!(dataSeries::RheoExperimentSeries, masked_indices::Vector{Int64})

modifies data by excluding data and corresponding features at selected masked_indices.
"""
function mask_out!(dataSeries::RheoExperimentSeries, masked_indices::Vector{Int64})
    bool = [i ∉ masked_indices for i in 1:length(dataSeries.data)]
    dataSeries.data = dataSeries.data[bool]
    for f in keys(dataSeries.features)
        dataSeries.features[f] = dataSeries.features[f][bool]
    end
end


"""
    mask_out(dataSeries::RheoExperimentSeries, masked_indices::Vector{Int64})

return a RheoExperimentSeries like dataSeries but excluding data and corresponding features at selected masked_indices.
"""
function mask_out(dataSeries::RheoExperimentSeries, masked_indices::Vector{Int64})
    bool_mask = [i ∉ masked_indices for i in 1:length(dataSeries.data)]
    f_dict = Dict()
    for f in keys(dataSeries.features)
        f_dict[f] = dataSeries.features[f][bool_mask]
    end
    return RheoExperimentSeries(dataSeries.data[bool_mask], f_dict)
end

function Base.length(x::RheoExperimentSeries)
    return length(x.data)
end

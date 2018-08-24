#!/usr/bin/env julia

#############################
#~ Preprocessing Functions ~#
#############################

"""
    var_resample(self::RheologyData, refvar::Symbol, pcntdownsample::Float64; mapback::Bool = false)

Convert a fixed sample rate array to a variable sample rate, with sampling points
added according to a relative change in chosen variable `refvar`, 1st derivative
of `refvar` and 2nd derivative of `refvar` (WRT time). Usually chosen as the
measured variable, so :σ for a stress relaxation test and :ϵ for a creep test.

Currently only variable downsampling supported. pcntdown sample is approximate,
works well in some cases and very poorly in others. If required, compare resampled
length vs original length after processing has finished. If data is noisy, may
benefit from sending smoothed signal to this algorithm and either using mapback
function or interpolating onto unsmoothed data.

See help docstring for `var_resample` for more details on algorithm implementation.
"""
function var_resample(self::RheologyData, refvar::Symbol, pcntdownsample::Float64; _mapback::Bool = false)

    # enforce minimum period of original period/10
    _minperiod = (self.t[2] - self.t[1])/10.0

    # get time resampled with respect to refvar
    (tᵦ, dummy)  = var_resample(self.t, getfield(self, refvar), pcntdownsample, _minperiod)

    local σ::Array{Float64,1}
    local ϵ::Array{Float64,1}
    local t::Array{Float64,1}

    if _mapback
        # get mapped indices wrt original
        mapback_indices = mapback(tᵦ, self.t)

        σ = self.σ[mapback_indices]
        ϵ = self.ϵ[mapback_indices]
        t = self.t[mapback_indices]
        
    elseif !_mapback
        # interpolate with respect to t
        σ_interp = Interpolations.interpolate((self.t,), self.σ, Interpolations.Gridded(Interpolations.Linear()))
        ϵ_interp = Interpolations.interpolate((self.t,), self.ϵ, Interpolations.Gridded(Interpolations.Linear()))
        t_interp = Interpolations.interpolate((self.t,), self.t, Interpolations.Gridded(Interpolations.Linear()))

        # resample all using new timepoints tᵦ
        σ = σ_interp[tᵦ]
        ϵ = ϵ_interp[tᵦ]
        t = t_interp[tᵦ]
    end

    # change sampling type to variable
    sampling = "variable"
    
    # add record of operation applied
    log = vcat(self.log, "var_resample - refvar: $refvar, pcntdownsample: $pcntdownsample, mapback: $_mapback")

    self_new = RheologyData(σ, ϵ, t, sampling, log)

end

"""
    downsample(self::RheologyData, time_boundaries::Array{Float64,1}, elperiods::Array{Int64,1})

High-level RheologyData interface to downsample in base.jl. Boundaries are floating point times which are then converted to the closest elements.
"""
function downsample(self::RheologyData, time_boundaries::Array{Float64,1}, elperiods::Array{Int64,1})

    # convert boundaries from times to element indicies
    boundaries = closestindices(self.t, time_boundaries)

    # get downsampled indices
    indices = downsample(boundaries::Array{Int64,1}, elperiods::Array{Int64,1})

    # downsample data
    σ = self.σ[indices]
    ϵ = self.ϵ[indices]
    t = self.t[indices]

    # change to variable sampling rate if more than one section, if not then as original
    local sampling::String
    if length(elperiods) > 1
        sampling = "variable"
    else
        sampling = self.sampling
    end

    # add record of operation applied
    log = vcat(self.log, "downsample - boundaries: $boundaries, elperiods: $elperiods")

    self_new = RheologyData(σ, ϵ, t, sampling, log)

end

"""
    fixed_resample(self::RheologyData, time_boundaries::Array{Float64,1}, elperiods::Array{Int64,1}, direction::Array{String,1})

High-level RheologyData interface to fixed_resample in base.jl
"""
function fixed_resample(self::RheologyData, time_boundaries::Array{Float64,1}, elperiods::Array{Int64,1}, direction::Array{String,1})

    # convert boundaries from times to element indicies
    boundaries = closestindices(self.t, time_boundaries)

    # resample all data
    (t, σ) = fixed_resample(self.t, self.σ, boundaries, elperiods, direction)
    (t, ϵ) = fixed_resample(self.t, self.ϵ, boundaries, elperiods, direction)

    # change to variable sampling rate if more than one section
    local sampling::String
    if length(elperiods) > 1
        sampling = "variable"
    else
        sampling = self.sampling
    end

    # add record of operation applied
    log = vcat(self.log, "fixed_resample - boundaries: $boundaries, elperiods: $elperiods, direction: $direction")

    self_new = RheologyData(σ, ϵ, t, sampling, log)

end

"""
    smooth(self::RheologyData, τ::Float64)

Smooth data using a Gaussian Kernel to time scale τ (approximately half power).

Smooths both σ and ϵ.
"""
function smooth(self::RheologyData, τ::Float64)

    @assert self.sampling=="constant" "Sample rate must be constant for Gaussian smoothing kernel to function properly"

    t = self.t

    # get constant sample period
    dt = t[2] - t[1]

    # sample rate
    samplerate = 1.0/dt;

    σ = smoothgauss(self.σ, τ, samplerate)
    ϵ = smoothgauss(self.ϵ, τ, samplerate)

    # add record of operation applied
    log = vcat(self.log, "smooth - τ: $τ")

    self_new = RheologyData(σ, ϵ, t, self.sampling, log)

end

"""
    function mapbackdata(self_new::RheologyData, self_original::RheologyData)

Map back elements (WRT closest time elements) of all data from self_new to
self_original. See mapback help docstring for more info on how algorithm works.
"""
function mapbackdata(self_new::RheologyData, self_original::RheologyData)

    # get mapped back indices
    indices = mapback(self_new.t, self_original.t)

    #  set variables
    σ = self_original.σ[indices]
    ϵ = self_original.ϵ[indices]
    t = self_original.t[indices]

    # add record of operation applied
    log = vcat(self_new.log, "mapped back") # mapped back to what? Include self_origina.log as well? Need to change Array type if so.

    # to add, check if sample rate is now variable or constant (unlikely but could fall back to constant?)

    self_new = RheologyData(σ, ϵ, t, self_new.sampling, log)

end

##########################
#~ Processing Functions ~#
##########################

"""
    modelfit(data::RheologyData, modulus::Function[, p0::Array{Float64,1}, lo::Array{Float64,1}, hi::Array{Float64,1}; verbose::Bool = false])

Fit RheologyData struct to model and return a fitted model as a RheologyModel object.

# Arguments

- `data`: RheologyData struct containing all data
- `modulus`: E.g. G_SLS, J_springpot etc. See base_models.jl for full list
- `p0`: Initial parameters to use in fit
- `lo`: Lower bounds for parameters
- `hi`: Higher bounds for parameters
- `verbose`: If true, prints parameters on each optimisation iteration
"""
function modelfit(data::RheologyData,
                  model::RheologyModel,
                  modtouse::Symbol;
                  p0::Array{Float64,1} = [-1.0],
                  lo::Array{Float64,1} = [-1.0],
                  hi::Array{Float64,1} = [-1.0],
                  verbose::Bool = false,
                  rel_tol = 1e-4)::RheologyModel

    # get modulus function
    modulus = getfield(model, modtouse)

    # use default p0 if not provided
    if quasinull(p0)
        p0 = model.parameters
    end 

    # get singularity presence
    sing = singularitytest(modulus, p0)

    # generate time series difference array (for convolution)
    dt_series = deriv(data.t)

    # get derivative of controlled variable and measured variable
    if modtouse == :J
        dcontrolled = deriv(data.σ, data.t)
        measured = data.ϵ
    elseif modtouse == :G
        dcontrolled = deriv(data.ϵ, data.t)
        measured = data.σ
    end

    # start fit
    tic()
    (minf, minx, ret) = leastsquares_init(p0,
                                          lo, 
                                          hi,
                                          modulus, 
                                          data.t, 
                                          dt_series, 
                                          dcontrolled,
                                          measured; 
                                          insight = verbose,
                                          sampling = data.sampling, 
                                          singularity = sing,
                                          _rel_tol = rel_tol)
    timetaken = toq()

    modulusname = string(modulus)

    log = vcat(data.log, "Fitted modulus $modulusname in $timetaken seconds, finished due to $ret with parameters $minx")

    RheologyModel(model.G, model.J, model.Gp, model.Gpp, minx, log)

end

"""
    modelpredict(data::RheologyData, model::RheologyModel)

Given data and model and parameters, predict new dataset based on both.
"""
function modelpredict(data::RheologyData, model::RheologyModel, modtouse::Symbol)::RheologyData

    # get modulus
    modulus = getfield(model, modtouse)

    # get singularity presence
    sing = singularitytest(modulus, model.parameters)

    if modtouse == :J
        dcontrolled = deriv(data.σ, data.t)
    elseif modtouse == :G
        dcontrolled = deriv(data.ϵ, data.t)
    end

    # generate time series difference array (for convolution)
    dt_series = deriv(data.t)

    # get convolution
    if !sing && data.sampling == "constant"
        convolved = boltzconvolve_nonsing(modulus, data.t, deriv(data.t), model.parameters, dcontrolled)

    elseif sing && data.sampling == "constant"
        convolved = boltzconvolve_sing(modulus, data.t, deriv(data.t), model.parameters, dcontrolled)

    elseif !sing && data.sampling == "variable"
        convolved = boltzintegral_nonsing(modulus, data.t, model.parameters, dcontrolled)

    elseif sing && data.sampling == "variable"
        convolved = boltzintegral_sing(modulus, data.t, model.parameters, dcontrolled)

    end

    if sing
        if modtouse == :J
            σ = data.σ[2:end]
            ϵ = convolved
        elseif modtouse == :G
            σ = convolved
            ϵ = data.ϵ[2:end]
        end   
        t = data.t[2:end]

    elseif !sing
        if modtouse == :J
            σ = data.σ
            ϵ = convolved
        elseif modtouse == :G
            σ = convolved
            ϵ = data.ϵ
        end 
        t = data.t

    end
        
    # store operation
    log = vcat(data.log, "Predicted data from model:", model.log)

    RheologyData(σ, ϵ, t, data.sampling, log)

end

function modelstepfit(data::RheologyData,
                  model::RheologyModel,
                  modtouse::Symbol;
                  p0::Array{Float64,1} = [-1.0],
                  lo::Array{Float64,1} = [-1.0],
                  hi::Array{Float64,1} = [-1.0],
                  verbose::Bool = false,
                  rel_tol = 1e-4)::RheologyModel

    # get modulus function
    modulus = getfield(model, modtouse)

    # use default p0 if not provided
    if quasinull(p0)
        p0 = model.parameters
    end 

    # get singularity presence
    sing = singularitytest(modulus, p0; t1 = data.t[1])

    # get controlled variable (just take first element as it's a step) and measured variable
    if modtouse == :J
        controlled = data.σ[1]
        measured = data.ϵ
    elseif modtouse == :G
        controlled = data.ϵ[1]
        measured = data.σ
    end

    # start fit
    tic()
    (minf, minx, ret) = leastsquares_stepinit(p0,
                                              lo, 
                                              hi,
                                              modulus, 
                                              data.t, 
                                              controlled,
                                              measured; 
                                              insight = verbose,
                                              singularity = sing,
                                              _rel_tol = rel_tol)
    timetaken = toq()

    modulusname = string(modulus)

    log = vcat(data.log, "Fitted modulus $modulusname in $timetaken seconds, finished due to $ret with parameters $minx")

    RheologyModel(model.G, model.J, model.Gp, model.Gpp, minx, log)

end

function modelsteppredict(data::RheologyData, model::RheologyModel, modtouse::Symbol; step_on::Real = 0.0)::RheologyData

    # get modulus
    modulus = getfield(model, modtouse)

    # get singularity presence
    sing = singularitytest(modulus, model.parameters; t1 = data.t[1])

    if modtouse == :J
        controlled = data.σ[1]
    elseif modtouse == :G
        controlled = data.ϵ[1]
    end

    # get convolution
    if !sing
        stepon_el = closestindex(data.t, step_on)
        predicted = zeros(length(data.t))
        predicted[(stepon_el):end] = controlled*modulus(data.t[stepon_el:end] - data.t[stepon_el], model.parameters)

    elseif sing
        stepon_el = closestindex(data.t, step_on)
        predicted = zeros(length(data.t))
        predicted[(stepon_el + 1):end] = controlled*modulus(data.t[(stepon_el + 1):end] - data.t[stepon_el], model.parameters)
    end

    if modtouse == :J
        σ = data.σ
        ϵ = predicted
    elseif modtouse == :G
        σ = predicted
        ϵ = data.ϵ
    end 
        
    # store operation
    log = vcat(data.log, "Predicted data from model:", model.log)

    RheologyData(σ, ϵ, data.t, data.sampling, log)

end

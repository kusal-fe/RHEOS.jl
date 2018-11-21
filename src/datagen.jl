#!/usr/bin/env julia

"""
    stepgen(t_total::Float64, t_on::Float64; t_trans::Float64 = (t_total - t_on)/100.0, amplitude::Float64 = 1.0, baseval::Float64 = 0.0, stepsize::Float64 = 1.0)

Generate RheologyData struct with a step function approximated by a logisitic function.
"""
function stepgen(t_total::Float64, 
                  t_on::Float64;
                  t_trans::Float64 = 0.0,
                  amplitude::Float64 = 1.0,
                  baseval::Float64 = 0.0,
                  stepsize::Float64 = 1.0)

    t = collect(0.0:stepsize:t_total)

    data = Array{Float64,1}(undef, length(t))
    # for smooth transition
    if t_trans>0.0
        k = 10.0/t_trans
        data = baseval .+ amplitude./(1 .+ exp.(-k*(t.-t_on)))
    # discrete jump
    elseif t_trans==0.0
        for (i, t_i) in enumerate(t)
            if t_i>=t_on
                data[i] = baseval + amplitude
            elseif t_i<t_on
                data[i] = baseval
            end
        end
    end

    RheologyData(data, t, ["stepgen: t_total: $t_total, t_on: $t_on, t_trans: $t_trans, amplitude: $amplitude, baseval: $baseval, stepsize: $stepsize"])

end

"""
    rampgen(t_total::Float64, t_start::Float64, t_stop::Float64; amplitude::Float64 = 1.0, baseval::Float64 = 0.0, stepsize::Float64 = 1.0)

Generate RheologyData struct with a ramp function.
"""
function rampgen(t_total::Float64,
                  t_start::Float64,
                  t_stop::Float64;
                  amplitude::Float64 = 1.0,
                  baseval::Float64 = 0.0,
                  stepsize::Float64 = 1.0)

    t = collect(0.0:stepsize:t_total)

    m = amplitude/(t_stop - t_start)

    data = baseval .+ zeros(length(t))

    for (i, v) in enumerate(t)

        if t_start<=v<t_stop
            data[i] += data[i-1] + stepsize*m
        elseif v>=t_stop
            data[i] = amplitude
        end

    end

    RheologyData(data, t, ["rampgen: t_total: $t_total, t_start: $t_start, t_stop: $t_stop, amplitude: $amplitude, baseval: $baseval, stepsize: $stepsize"])

end

"""
    singen(t_total::Float64, frequency::Float64; t_start::Float64 = 0.0, phase::Float64 = 0.0, amplitude::Float64 = 1.0, baseval::Float64 = 0.0, stepsize::Float64 = 1.0)

Generate RheologyData struct with a sinusoidal function.
"""
function singen(t_total::Float64,
                frequency::Float64;
                t_start::Float64 = 0.0,
                phase::Float64 = 0.0,
                amplitude::Float64 = 1.0,
                baseval::Float64 = 0.0,
                stepsize::Float64 = 1.0)

    t = collect(0.0:stepsize:t_total)

    data = baseval .+ zeros(length(t))

    for (i, v) in enumerate(t)

        if v>=t_start
            data[i] += amplitude*sin(2*π*frequency*(v - t_start) + phase)
        end

    end

    RheologyData(data, t, ["singen: t_total: $t_total, frequency: $frequency, t_start: $t_start, phase: $phase, amplitude: $amplitude, baseval: $baseval, stepsize: $stepsize"])

end

"""
    repeatdata(self::RheologyData, n::Integer)

Repeat a given RheologyData generated data set n times.
"""
function repeatdata(self::RheologyData, n::Integer; t_trans = 0.0)

    @assert self.σ==self.ϵ "Repeat data only works when σ==ϵ which is the state of RheologyData after being generated using the built-in RHEOS data generation functions"
    
    dataraw = self.σ

    @assert constantcheck(self.t) "Data sample-rate must be constant"

    step_size = self.t[2] - self.t[1]

    t = collect(0.0:step_size:(self.t[end]*n))

    # smooth transition between repeats
    if t_trans>0.0
        elbuffer = round(Int, (1/2)*(t_trans/step_size))
        selflength = length(self.t)

        # ensure smooth transition from end of data to new beginning so no discontinuities
        data_smooth_end = stepgen(self.t[end], self.t[end]; t_trans = t_trans, amplitude = dataraw[1] - dataraw[end], baseval = dataraw[selflength - elbuffer], stepsize = step_size)
        data_smooth_start = stepgen(self.t[end], 0.0; t_trans = t_trans, amplitude = dataraw[1] - dataraw[end], baseval = dataraw[selflength - elbuffer], stepsize = step_size)
        data_smoother = data_smooth_end + data_smooth_start

        data_single = self + data_smoother

        data = repeat(data_single.data[1:end] - dataraw[selflength - elbuffer], outer=[n])

        for i = 1:(n-1)
            deleteat!(data, i*length(self.t) + (2-i))
        end

        # fix first repeat
        for (i, v) in enumerate(dataraw)
            if i<(elbuffer*100) && i<round(Int, length(self.t)/2) 
                data[i] = v
            end
        end

        log = vcat(self.log, ["repeated data $n times with transition time $t_trans"])

        return RheologyData(data, t, log)

    # discrete jump
    elseif t_trans==0.0

        data = repeat(dataraw, outer=[n])

        for i = 1:(n-1)
            deleteat!(data, i*length(self.t) + (2-i))
        end

        log = vcat(self.log, ["repeated data $n times with transition time $t_trans"])

        return RheologyData(data, t, log)

    end

end

"""
    addnoise(self::RheologyData; amplitude::Float64 = 0.1, seed::Union{Int, Nothing} = nothing)

Add random noise to artificially generated data.
"""
function addnoise(self::RheologyData; amplitude::Float64 = 0.1, seed::Union{Int, Nothing} = nothing)

    # conditional loading of random as this is the only RHEOS function which requires it
    @eval import Random: seed!, rand

    @assert self.σ==self.ϵ "addnoise only works when σ==ϵ which is the state of RheologyData after being generated using the built-in RHEOS data generation functions"

    dataraw = self.σ

    if typeof(seed)==Nothing
        # get random seed
        seed!()
    elseif typeof(seed)<:Int
        # use specified seed
        @assert seed>=0 "Seed must be non-negative"
        seed!(seed)
    end

    data = dataraw .+ amplitude*(2*rand(Float64, length(dataraw)) .- 1)

    log = vcat(self.log, ["added noise of amplitude $amplitude, with seed $seed"])

    RheologyData(data, self.t, log)

end


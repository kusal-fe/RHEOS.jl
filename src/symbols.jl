#!/usr/bin/env julia

# This file contains data and functions helping with the translation of symbols for the use of RHEOS without unicode symbols.


# translation table for symbols. left = symbol use by user, right = symbol used by model declaraction or RHEOS structs.
const symbol_convertion_table=(
    time = :t,
    strain = :ϵ,
    epsilon = :ϵ,
    ε = :ϵ,
    stress = :σ,
    sigma = :σ,
    omega = :ω,
    eta = :η,
    alpha = :α,
    beta = :β,
    c_alpha = :cₐ,
    c_beta= :cᵦ,
    )


function symbol_to_unicode(s::Symbol)    
	s in keys(symbol_convertion_table) ? symbol_convertion_table[s] : s 
	end

function symbol_to_unicode(nt)    
	NamedTuple{Tuple([ symbol_to_unicode(s) for s in keys(nt) ])}( values(nt) )
	end





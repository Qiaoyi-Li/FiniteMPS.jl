"""
     abstract type SweepDirection end

Direction of sweeps, wrapped as a type to be passed as a type parameter. 
"""
abstract type SweepDirection end

"""
     struct SweepL2R <: SweepDirection end
"""
struct SweepL2R <: SweepDirection end

"""
     struct SweepR2L <: SweepDirection end
"""
struct SweepR2L <: SweepDirection end

"""
     struct AnyDirection <: SweepDirection end
"""
struct AnyDirection <: SweepDirection end
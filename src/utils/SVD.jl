module InplaceSVD

using LinearAlgebra.LAPACK: require_one_based_indexing, chkstride1, BlasInt, chklapackerror, libblastrampoline

export inplace_gesdd!, inplace_gesvd!

# modify LAPACK.gesdd!('S', A) to use preallocated U, S, VT
function inplace_gesdd!(U::AbstractMatrix{T},
	S::AbstractVector{Float64},
	VT::AbstractMatrix{T},
	A::AbstractMatrix{T}) where T <: Union{Float64, ComplexF64}

	require_one_based_indexing(A)
	chkstride1(A)
	m, n  = size(A)
	minmn = min(m, n)

	work  = Vector{T}(undef, 1)
	lwork = BlasInt(-1)
	cmplx = eltype(A) <: Complex
	if cmplx
		rwork = Vector{Float64}(undef, minmn * max(5 * minmn + 7, 2 * max(m, n) + 2 * minmn + 1))
	end
	iwork = Vector{BlasInt}(undef, 8 * minmn)
	info  = Ref{BlasInt}()
	for i ∈ 1:2  # first call returns lwork as work[1]
		if cmplx # assume 
			ccall((:zgesdd_64_, libblastrampoline), Cvoid,
				(Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T},
					Ref{BlasInt}, Ptr{Float64}, Ptr{T}, Ref{BlasInt},
					Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt},
					Ptr{Float64}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
				'S', m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
				work, lwork, rwork, iwork, info, 1)
		else
			ccall((:dgesdd_64_, libblastrampoline), Cvoid,
				(Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{T},
					Ref{BlasInt}, Ptr{T}, Ptr{T}, Ref{BlasInt},
					Ptr{T}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt},
					Ptr{BlasInt}, Ptr{BlasInt}, Clong),
				'S', m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
				work, lwork, iwork, info, 1)
		end
		chklapackerror(info[])
		if i == 1
			# Work around issue with truncated Float32 representation of lwork in
			# sgesdd by using nextfloat. See
			# http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
			# and
			# https://github.com/scipy/scipy/issues/5401
			lwork = round(BlasInt, nextfloat(real(work[1])))
			resize!(work, lwork)
		end
	end
	return nothing
end

# modify LAPACK.gesvd!('S', 'S', A) to use preallocated U, S, VT
function inplace_gesvd!(U::AbstractMatrix{T},
	S::AbstractVector{Float64},
	VT::AbstractMatrix{T},
	A::AbstractMatrix{T}) where T <: Union{Float64, ComplexF64}

	require_one_based_indexing(A)
	chkstride1(A)

	m, n  = size(A)
	minmn = min(m, n)

	work  = Vector{T}(undef, 1)
	cmplx = eltype(A) <: Complex
	if cmplx
		rwork = Vector{Float64}(undef, 5minmn)
	end
	lwork = BlasInt(-1)
	info  = Ref{BlasInt}()
	for i in 1:2  # first call returns lwork as work[1]
		if cmplx
			ccall((:zgesvd_64_, libblastrampoline), Cvoid,
				(Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
					Ptr{T}, Ref{BlasInt}, Ptr{Float64}, Ptr{T},
					Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T},
					Ref{BlasInt}, Ptr{Float64}, Ptr{BlasInt}, Clong, Clong),
				'S', 'S', m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
				work, lwork, rwork, info, 1, 1)
		else
			ccall((:dgesvd_64_, libblastrampoline), Cvoid,
				(Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
					Ptr{T}, Ref{BlasInt}, Ptr{T}, Ptr{T},
					Ref{BlasInt}, Ptr{T}, Ref{BlasInt}, Ptr{T},
					Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
				'S', 'S', m, n, A, max(1, stride(A, 2)), S, U, max(1, stride(U, 2)), VT, max(1, stride(VT, 2)),
				work, lwork, info, 1, 1)
		end
		chklapackerror(info[])
		if i == 1
			lwork = BlasInt(real(work[1]))
			resize!(work, lwork)
		end
	end

	return nothing

end

end

using .InplaceSVD

function inplace_svd!(
	U::AbstractMatrix{T},
	S::AbstractVector{Float64},
	VT::AbstractMatrix{T},
	A::AbstractMatrix{T},
	::SVD) where T <: Union{Float64, ComplexF64}
	return inplace_gesvd!(U, S, VT, A)
end
function inplace_svd!(
	U::AbstractMatrix{T},
	S::AbstractVector{Float64},
	VT::AbstractMatrix{T},
	A::AbstractMatrix{T},
	::SDD) where T <: Union{Float64, ComplexF64}
	return inplace_gesdd!(U, S, VT, A)
end

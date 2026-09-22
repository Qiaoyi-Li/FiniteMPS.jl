function _nearidentity(rng::Random.AbstractRNG, ::Type{T}, V, σ::Real) where {T}
     A = TensorMap{T}(undef, V ← V)
     for (_, b) in blocks(A)
          n = size(b, 1)
          g = zeros(T, n, n)
          for i in 1:n, j in i+1:n
               g[i, j] = randn(rng, T) * σ
               g[j, i] = -conj(g[i, j])
          end
          copyto!(b, exp(g))
     end
     return A
end

"""
	calObs!(Tree::ObservableTree{L},
		Ψ::AbstractMPS{L},
		Φ::AbstractMPS{L} = Ψ;
		kwargs...
	) -> TO::TimerOutput

Calculate observables stored in `Tree`, using bra `⟨Ψ|` and ket `|Φ⟩`.

# Kwargs 
	El::AbstractTensorMap 
	Er::AbstractTensorMap 
Manually set boundary left or right environment tensor. Default is the `rank-2` isometry deduced from `Ψ` and `Φ`, which may be incorrect if the operators have nontrivial auxiliary spaces. 

	normalize::Bool = false 
If 'true', calculate `⟨Ψ|O|Φ⟩/(|Ψ||Φ|)` instead of `⟨Ψ|O|Φ⟩`.

	serial::Bool = false
Force to choose serial mode, usually used when debugging.

	disk::Bool = false 
Store environment tensor in disk or not.

	maxsize::Int64 = maximum(treewidth(Tree))
Number of environment tensors left in memory. Default is the larger one of the width of left tree or right tree. 

	verbose::Int64 = 0
Show the timer several times if `verbose > 0`.

	showtimes::Int64 = 10 
Times to show the timer.

	GCSpacing::Int64 = 100
The spacing (count nodes) between two nearest manual `GC.gc()`.

	ntasks::Int64 = get_num_threads()
Number of distributed tasks, used in multithreaded mode.
"""
function calObs!(Tree::ObservableTree{L}, Ψ::AbstractMPS{L}, Φ::AbstractMPS{L} = Ψ; kwargs...) where L

	disk::Bool = get(kwargs, :disk, false)
	verbose::Int64 = get(kwargs, :verbose, 0)

	# make sure the tree is merged
	merge!(Tree)

	width = treewidth(Tree)
	if verbose > 0
		println("Tree size = ($(treesize(Tree.RootL)), $(treesize(Tree.RootR))), width = $(width)!")
	end

	# reset refs 
	for d in values(Tree.Refs), (k, v) in d
		v[] = NaN
	end

	# generate a map from nodes to [status, filename]
	nodemap = Dict{InteractionTreeNode, Dict{Symbol, Any}}()
	if disk
		dir = mktempdir()
	end
	n_count = 0
	for R in [Tree.RootL, Tree.RootR], node in PreOrderDFS(R)
		si = node.Op[1]
		# i is unique for each node
		filename = disk ? joinpath(dir, "node_$(si)_$(n_count).bin") : ""
		nodemap[node] = Dict{Symbol, Any}(:st => false, :filename => filename)
		n_count += 1
	end

	# two caches
	if disk
		# default size = max treewidth
		maxsize = get(kwargs, :maxsize, maximum(width))
	else
		maxsize = n_count
	end
	# finalizer to store in disk
	function f(k, v)
		filename = nodemap[k][:filename]
		!isempty(filename) && serialize(filename, v)
		# TODO test Ptr array 
		return nothing
	end
	cacheL = LRU{InteractionTreeNode, LocalLeftTensor}(maxsize = maxsize, finalizer = f)
	cacheR = LRU{InteractionTreeNode, LocalRightTensor}(maxsize = maxsize, finalizer = f)

	# initialize boundary environment tensors
	El = get(kwargs, :El, _defaultEl(Ψ, Φ))
	Er = get(kwargs, :Er, _defaultEr(Ψ, Φ))
	
	cacheL[Tree.RootL] = El
	nodemap[Tree.RootL][:st] = true
	cacheR[Tree.RootR] = Er
	nodemap[Tree.RootR][:st] = true

	if get(kwargs, :serial, false) || get_num_threads_julia() == 1
		TO = _calObs_serial!(Tree, Ψ, Φ, nodemap, cacheL, cacheR; kwargs...)
	else
		TO = _calObs_threading!(Tree, Ψ, Φ, nodemap, cacheL, cacheR; kwargs...)
	end

	# check refs 
	for d in values(Tree.Refs), (k, v) in d
		@assert !isnan(v[])
	end

	# deal with normalization factor, note current values do not contain the normalization factor
	if !get(kwargs, :normalize, false)
		fac = coef(Ψ) * coef(Φ)
		for d in values(Tree.Refs), v in values(d)
			v[] *= fac
		end
	end
		

	# cleanup 
	for c in [cacheL, cacheR]
		# empty filename to avoid unnecessary serialization
		for (k, v) in c
			nodemap[k][:filename] = ""
		end
		empty!(c.dict)
	end
	if disk
		rm(dir; recursive = true, force = true)
	end

	manualGC()

	return TO
end

function _calObs_serial!(Tree::ObservableTree{L},
	Ψ::AbstractMPS{L},
	Φ::AbstractMPS{L},
	nodemap::Dict{InteractionTreeNode, Dict{Symbol, Any}},
	cacheL::LRU{InteractionTreeNode, LocalLeftTensor},
	cacheR::LRU{InteractionTreeNode, LocalRightTensor};
	kwargs...) where L

	GCspacing::Int64 = get(kwargs, :GCspacing, 100)
	verbose::Int64 = get(kwargs, :verbose, 0)
	showtimes::Int64 = get(kwargs, :showtimes, 10)

	showspacing::Int64 = cld(length(nodemap), showtimes)

	TO = TimerOutput()
	n_count = 0

	# finish right tree first
	@timeit TO "right tree" begin
		for node in StatelessBFS(Tree.RootR)
			si = node.Op[1] - 1
			# get Er
			@timeit TO "load Er" begin
				if haskey(cacheR, node)
					Er = cacheR[node]
				else
					# load from disk and store in cache
					Er = deserialize(nodemap[node][:filename])
					cacheR[node] = Er
				end
			end

			# process all children
			for ch in node.children
				Op = deepcopy(Tree.Ops[si][ch.Op[2]])
				Op.strength[] = 1.0
				@timeit TO "pushleft" Er_ch = _pushleft(Er, Ψ[si]', Op, Φ[si])
				@timeit TO "save Er" cacheR[ch] = Er_ch
			end

			nodemap[node][:st] = true

			# print
			n_count += 1
			if iszero(n_count % GCspacing)
				manualGC(TO)
			end
			if verbose > 0 && n_count % showspacing == length(nodemap) % showspacing
				show(TO; title = "$(n_count) / $(length(nodemap))")
				println()
				flush(stdout)
			end

		end
	end

	# left tree
	@timeit TO "left tree" begin
		for node in StatelessBFS(Tree.RootL)
			si = node.Op[1] + 1
			# get El
			@timeit TO "load El" begin
				if haskey(cacheL, node)
					El = cacheL[node]
				else
					El = deserialize(nodemap[node][:filename])
					cacheL[node] = El
				end
			end

			# process all children
			for ch in node.children
				Op = deepcopy(Tree.Ops[si][ch.Op[2]])
				Op.strength[] = 1.0
				@timeit TO "pushright" El_ch = _pushright(El, Ψ[si]', Op, Φ[si])
				@timeit TO "save El" cacheL[ch] = El_ch
			end

			# process all Intrs 
			# deal with duplicate channels
			node2val = Dict{InteractionTreeNode, Number}()
			for ch in node.Intrs
				node_R = ch.LeafR
				if haskey(node2val, node_R)
					ch.ref[] = node2val[node_R]
					continue
				end
				@timeit TO "load Er" begin
					if haskey(cacheR, node_R)
						Er = cacheR[node_R]
					else
						Er = deserialize(nodemap[node_R][:filename])
						cacheR[node_R] = Er
					end
				end
				@timeit TO "trace" val = El * Er
				node2val[node_R] = ch.ref[] = val
			end

			# El is no longer needed
			@timeit TO "cleanup" begin
				rm(nodemap[node][:filename]; force = true)
				nodemap[node][:filename] = ""
				delete!(cacheL, node)
			end

			nodemap[node][:st] = true

			# print
			n_count += 1
			if iszero(n_count % GCspacing)
				manualGC(TO)
			end
			if verbose > 0 && n_count < length(nodemap) && n_count % showspacing == length(nodemap) % showspacing
				show(TO; title = "$(n_count) / $(length(nodemap))")
				println()
				flush(stdout)
			end
		end
	end

	if verbose > 0
		show(TO; title = "$(n_count) / $(length(nodemap))")
		println()
		flush(stdout)
	end

	return TO
end

function _calObs_threading!(Tree::ObservableTree{L},
	Ψ::AbstractMPS{L},
	Φ::AbstractMPS{L},
	nodemap::Dict{InteractionTreeNode, Dict{Symbol, Any}},
	cacheL::LRU{InteractionTreeNode, LocalLeftTensor},
	cacheR::LRU{InteractionTreeNode, LocalRightTensor};
	kwargs...) where L

	ntasks::Int64 = get(kwargs, :ntasks, get_num_threads_julia())
	GCspacing::Int64 = get(kwargs, :GCspacing, 100)
	verbose::Int64 = get(kwargs, :verbose, 0)
	showtimes::Int64 = get(kwargs, :showtimes, 10)

	showspacing::Int64 = cld(length(nodemap), showtimes)

	TO = TimerOutput()

	# finish right tree first 
	N_R = treesize(Tree.RootR)
	NodePool = Channel{InteractionTreeNode}(Inf)
	Ch_swap = Channel{Any}(Inf) # [node, TO] or error massage
	tasks = map(1:ntasks-1) do _
		Threads.@spawn begin
			while isopen(NodePool)
				TO_local = TimerOutput()

				@timeit TO_local "take" node = take!(NodePool)
				si = node.Op[1] - 1

				try
					# get Er 
					@timeit TO_local "load Er" Er = _getindex_disk(cacheR, node, x -> deserialize(nodemap[x][:filename]))

					# process all children
					for ch in node.children
						Op = deepcopy(Tree.Ops[si][ch.Op[2]])
						Op.strength[] = 1.0
						@timeit TO_local "pushleft" Er_ch = _pushleft(Er, Ψ[si]', Op, Φ[si])
						@timeit TO_local "save Er" _setindex_disk!(cacheR, Er_ch, ch)

						# push to swap channel 
						put!(Ch_swap, ch)
					end

					nodemap[node][:st] = true
					put!(Ch_swap, (node, TO_local))

				catch e
					put!(Ch_swap, e)
				end

			end
		end
	end
	# initialize recursion
	put!(NodePool, Tree.RootR)

	# swap 
	i = 0
	@timeit TO "right tree" begin
		while i < N_R
			rslt = take!(Ch_swap)
			if isa(rslt, Exception)
				throw(rslt)
			elseif isa(rslt, Tuple) # finished node
				_, TO_local = rslt
				# merge TO 
				merge!(TO, TO_local; tree_point = [TO.prev_timer_label])
				i += 1

				# GC and print 
				if i % GCspacing == 0
					manualGC(TO)
				end
				if verbose > 0 && i % showspacing == length(nodemap) % showspacing
					show(TO; title = "$(i) / $(length(nodemap))")
					println()
					flush(stdout)
				end
			else # child node
				put!(NodePool, rslt)
			end

		end
	end

	# kill the tasks
	close(NodePool)

	# left tree
	N_L = treesize(Tree.RootL)
	NodePool = Channel{InteractionTreeNode}(Inf)
	tasks = map(1:ntasks-1) do _
		Threads.@spawn begin
			while isopen(NodePool)
				TO_local = TimerOutput()
				@timeit TO_local "take" node = take!(NodePool)
				si = node.Op[1] + 1
				try
					# get El, note El is no longer needed
					@timeit TO_local "load El" El = _getindex_disk(cacheL, node, x -> deserialize(nodemap[x][:filename]); update = false)

					# clean 
					@timeit TO_local "cleanup" begin
						rm(nodemap[node][:filename]; force = true)
						nodemap[node][:filename] = ""
					end

					# process all children
					for ch in node.children
						Op = deepcopy(Tree.Ops[si][ch.Op[2]])
						Op.strength[] = 1.0
						@timeit TO_local "pushright" El_ch = _pushright(El, Ψ[si]', Op, Φ[si])
						@timeit TO_local "save El" _setindex_disk!(cacheL, El_ch, ch)

						# push to swap channel
						put!(Ch_swap, ch)
					end

					# process all Intrs
					# deal with duplicate channels
					node2val = Dict{InteractionTreeNode, Number}()
					for ch in node.Intrs
						node_R = ch.LeafR
						if haskey(node2val, node_R)
							ch.ref[] = node2val[node_R]
							continue
						end
						# load Er 
						@timeit TO_local "load Er" Er = _getindex_disk(cacheR, node_R, x -> deserialize(nodemap[x][:filename]))

						@timeit TO_local "trace" node2val[node_R] = ch.ref[] = El * Er
					end


					nodemap[node][:st] = true
					put!(Ch_swap, (node, TO_local))

				catch e
					put!(Ch_swap, e)
				end
			end
		end

	end

	# initialize recursion
	put!(NodePool, Tree.RootL)

	# swap 
	i = 0
	@timeit TO "left tree" begin
		while i < N_L
			rslt = take!(Ch_swap)
			if isa(rslt, Exception)
				throw(rslt)
			elseif isa(rslt, Tuple) # finished node
				_, TO_local = rslt
				# merge TO 
				merge!(TO, TO_local; tree_point = [TO.prev_timer_label])
				i += 1

				# GC and print
				if (i + N_R) % GCspacing == 0
					manualGC(TO)
				end
				if verbose > 0 && (i < N_L) && (i + N_R) % showspacing == length(nodemap) % showspacing
					show(TO; title = "$(i + N_R) / $(length(nodemap))")
					println()
					flush(stdout)
				end

			else # child node
				put!(NodePool, rslt)
			end

		end
	end

	if verbose > 0
		show(TO; title = "$(i + N_R) / $(length(nodemap))")
		println()
		flush(stdout)
	end

	# kill the tasks
	close(NodePool)

	return TO
end

function _getindex_disk(lru::LRU{K, V}, key, f; update::Bool = true) where {K, V}
	# modified from LRUCache.getindex to support threading-safe loading from disk via a function f
	lock(lru.lock) do
		if LRUCache._unsafe_haskey(lru, key)
			v, n, s = lru.dict[key]
			LRUCache._move_to_front!(lru.keyset, n)
			if !update
				# delete the key 
				lru.currentsize -= s
				LRUCache._delete!(lru.keyset, n)
				delete!(lru.dict, key)
			end
			return v
		else
			v = try
				f(key)
			catch
				rethrow()
			end
			# ======== copy from LRUCache.setindex! ==========
			if update
				evictions = Tuple{K, V}[]
				LRUCache._unsafe_addindex!(lru, v, key)
				LRUCache._unsafe_resize!(lru, evictions)
				LRUCache._finalize_evictions!(lru.finalizer, evictions)
			end
			# ------------------------------------------
			return v
		end
	end
end

function _setindex_disk!(lru::LRU{K, V}, v, key) where {K, V}
	# modified from LRUCache.setindex! to make finalizer threading-safe 
	evictions = Tuple{K, V}[]
	lock(lru.lock) do
		if LRUCache._unsafe_haskey(lru, key)
			old_v, n, s = lru.dict[key]

			lru.currentsize -= s
			s = lru.by(v)::Int
			# If new entry is larger than entire cache, don't add it
			# (but still evict the old entry!)
			if s > lru.maxsize
				# We are inside the lock still, so we will remove it manually rather than
				# `delete!(lru, key)` which would need the lock again.
				delete!(lru.dict, key)
				LRUCache._delete!(lru.keyset, n)
			else # add the new entry
				lru.currentsize += s
				lru.dict[key] = (v, n, s)
				LRUCache._move_to_front!(lru.keyset, n)
			end
		else
			LRUCache._unsafe_addindex!(lru, v, key)
		end
		LRUCache._unsafe_resize!(lru, evictions)
		LRUCache._finalize_evictions!(lru.finalizer, evictions)
	end
	return lru
end

function _defaultEl(Ψ::AbstractMPS{L}, Φ::AbstractMPS{L}) where L 
	# default case, no horizontal bond
	v1 = codomain(Ψ[1])[1]
	v2 = codomain(Φ[1])[1]
	return v1 == v2 ? id(v1) : nothing
end
function _defaultEr(Ψ::AbstractMPS{L}, Φ::AbstractMPS{L}) where L 
	# default case, no horizontal bond
	v1 = domain(Φ[end])[end]
	v2 = domain(Ψ[end])[end]
	return v1 == v2 ? id(v1) : nothing
end

	
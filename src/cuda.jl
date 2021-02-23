## CUDA methods.

using CUDA

const BLCK = (256, 1)


function simcoarser(g, a::CuArray{T, N}) where {T, N}
    CUDA.zeros(eltype(a), ntuple(i->(size(a, i) - 2g) ÷ 2 + 2g, Val(N)))
end


function simfiner(g, a::CuArray{T, N}) where {T, N}
    CUDA.zeros(eltype(a), ntuple(i->(size(a, i) - 2g) * 2 + 2g, Val(N)))
end


function blocks(n, bsize)
    return div(n, bsize, RoundUp)
end


function blocks(g, a::AbstractArray, bsizes)
    blocks.(size(a) .- 2g, bsizes)
end


"""
   Obtain a CartesianIndex from the thread and block coordinates.  Needs an
   array `a` which is used only to specialize on dimension
"""
@inline @generated function cudaindex(g, a::AbstractArray{T, N}) where {T, N}
    if N == 2
        expr = quote
            i = g + (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = g + (blockIdx().y - 1) * blockDim().y + threadIdx().y
            
            return CartesianIndex(i, j)
        end
    elseif N == 3
        expr = quote
            i = g + (blockIdx().x - 1) * blockDim().x + threadIdx().x
            j = g + (blockIdx().y - 1) * blockDim().y + threadIdx().y
            k = g + (blockIdx().z - 1) * blockDim().z + threadIdx().z
            return CartesianIndex(i, j, k)
        end
    else
        throw(ArgumentError("Only arrays with 2, 3 dimensions allowed"))
    end
    expr       
end

"""
    Check that the given index is inside the valid part of the array with
    `g` ghost cells on each side
"""
@inline function cudainside(g, a::AbstractArray{T, N}, ind) where {T, N}
    all(ntuple(i->((firstindex(a, i) + g) <= ind[i] <= (lastindex(a, i) - g)),
               Val(N)))
end


function redblack2(f, g, a::CuArray, parity)
    function kern()
        i = g + 2 * ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) + 1
        j = g + (blockIdx().y - 1) * blockDim().y + threadIdx().y
        
        p = xor(parity, iseven(j - g))
        ind = CartesianIndex(i + p, j)
        cudainside(g, a, ind) || return nothing
        
        f(ind)
    
        nothing
    end

    @cuda(threads=BLCK,
          blocks=blocks.(rbends(g, a), BLCK),
          kern())

    nothing
end



function residual!(g, r::CuArray, u::CuArray, b::CuArray, s, c::AbstractConnector)
    st = lplstencil(u)
    
    function kern()
        ind = cudaindex(g, r)
        cudainside(g, r, ind) || return nothing
        
        l = laplacian(g, u, ind, st, c)
        @inbounds r[ind] = s * b[ind] + l
        
        nothing
    end

    @cuda(threads=BLCK,
          blocks=blocks.(inends(g, r), BLCK),
          kern())

    nothing
end


function restrict!(g, rh::CuArray, r::CuArray)
    st = cubestencil(r)
    
    function kern()
        irh = cudaindex(g, rh)
        cudainside(g, rh, irh) || return nothing
        
        ir = 2 * (irh - (g + 1) * oneunit(irh)) + (g + 1) * oneunit(irh)
        s = zero(eltype(r))
    
        for j in st
            @inbounds s += r[ir + j]
        end
        
        @inbounds rh[irh] = 4 * s / length(st)
    
        nothing
    end

    @cuda(threads=BLCK,
          blocks=blocks.(inends(g, rh), BLCK),
          kern())

    nothing
end


function interpolate!(g, r::CuArray, rh::CuArray, update::Type{Val{V}}=Val{false}) where {V}
    st = cubestencil(r)
    weights = binterpweights(st)
        
    function kern()
        irh = cudaindex(g, r)
        cudainside(g, rh, irh) || return nothing
        
        ir = 2 * (irh - (g + 1) * oneunit(irh)) + (g + 1) * oneunit(irh)
        for s in st
            # We will compute the value of cell in F at this location
            indf = ir + s
            
            # Delta to the furthest cell in H that plays into the interpolation
            δ = 2 * s - oneunit(irh)
            
            # Now we run over cells in H to compute the interpolation
            s = zero(eltype(r))
            for (w, sh) in zip(weights, st)
                @inbounds s += w * rh[irh + CartesianIndex(Tuple(δ) .* Tuple(sh))]
            end
            if V
                @inbounds r[indf] += s
            else
                @inbounds r[indf] = s
            end
        end
        
        nothing
    end


    @cuda(threads=BLCK,
          blocks=blocks.(inends(g, rh), BLCK),
          kern())

    nothing
end



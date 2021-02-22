using CUDA

# Arrays supported in the device
const CuSuppArray = Union{CuArray, OffsetArray{T, N, Q}  where {T, N, Q<:CuArray}}
BLCK = (256, 1)


function simcoarser(a::CuSuppArray)
    s = innersize(a)
    ndims = map(n->0:(n ÷ 2 + 1), s)
    
    p = CUDA.zeros(eltype(a), length.(ndims))
    OffsetArray(p, ndims...)
end


function simfiner(a::CuSuppArray)
    s = innersize(a)
    ndims = map(n->0:(n * 2 + 1), s)

    p = CUDA.zeros(eltype(a), length.(ndims))
    OffsetArray(p, ndims...)    
end


function inends(a::AbstractArray{T, N}) where {T, N}
    l = lastindex.(axes(a))
    t = ntuple(n -> (l[n] - 1), Val(N))

    t
end

function rbends(a::AbstractArray{T, N}) where {T, N}
    l = lastindex.(axes(a))
    t = ntuple(n -> (l[n + 1] - 1), Val(N - 1))

    @assert iseven(l[1] - 1)
    h = div(l[1] - 1, 2)

    (h, t...)
end


function blocks(n, bsize)
    return div(n, bsize, RoundUp)
end


function blocks(a::AbstractArray, bsizes)
    blocks.(size(a), bsizes)
end


# ONLY 2D
@inline function cudaindex()
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    CartesianIndex(i, j)
end

# ONLY 2D
@inline function cudainside(a, ind)
    (ind[1] <= lastindex(a, 1) - 1 && ind[2] <= lastindex(a, 2) - 1)
end


function redblack2(f, a::CuSuppArray, parity) where {T, N, Q<:CuArray}
    rng = rbends(a)

    @cuda(threads=BLCK,
          blocks=blocks.(rbends(a), BLCK),
          kern_redblack2(f, a, parity))

    nothing
end

function kern_redblack2(f, a, parity)
    i = 2 * ((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1) + 1
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    
    p = xor(parity, iseven(j))
    
    (i + p > lastindex(a, 1) - 1 || j > lastindex(a, 2) - 1) && return nothing
    
    f(CartesianIndex((i + p, j)))
    
    nothing
end


function residual!(r::CuSuppArray, u::CuSuppArray, b::CuSuppArray, s, c::AbstractConnector)
    st = lplstencil(u)

    @cuda(threads=BLCK,
          blocks=blocks.(inends(r), BLCK),
          kern_residual!(r, u, b, s, c, st))

    nothing
end

function kern_residual!(r, u, b, s, c, st)
    ind = cudaindex()
    cudainside(r, ind) || return nothing
    
    l = laplacian(u, ind, st, c)
    r[ind] = s * b[ind] + l
    
    nothing
end


function restrict!(rh::CuSuppArray, r::CuSuppArray)
    st = cubestencil(r)
    
    @cuda(threads=BLCK,
          blocks=blocks.(inends(rh), BLCK),
          kern_restrict!(rh, r, st))

    nothing
end


function kern_restrict!(rh, r, st)
    irh = cudaindex()
    cudainside(rh, irh) || return nothing
    
    ir = 2 * (irh - oneunit(irh)) + oneunit(irh)
    s = zero(eltype(r))
    
    for j in st
        @inbounds s += r[ir + j]
    end
    
    @inbounds rh[irh] = 4 * s / length(st)
    
    nothing
end


function interpolate!(r::CuSuppArray, rh::CuSuppArray, update::Type{Val{V}}=Val{false}) where {V}
    st = cubestencil(r)
    weights = binterpweights(st)
        
    @cuda(threads=BLCK,
          blocks=blocks.(inends(rh), BLCK),
          kern_interpolate!(r, rh, update, st, weights))

    nothing
end


function kern_interpolate!(r, rh, update::Type{Val{V}}, st, weights) where {V}
    irh = cudaindex()
    cudainside(rh, irh) || return nothing
    
    ir = 2 * (irh - oneunit(irh)) + oneunit(irh)
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
            r[indf] += s
        else
            r[indf] = s
        end
    end
    
    nothing
end

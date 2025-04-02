## KernelAbstractions methods.

const BLCK = (256, 1)

# Later we can remove this and make it more general.
using Metal
#const DeviceArray{T, N} = Union{CuArray{T, N}, MtlArray{T, N}}
const DeviceArray{T, N} = MtlArray{T, N}

function simcoarser(g, a::DeviceArray{T, N}) where {T, N}
    KA.zeros(get_backend(a), eltype(a), ntuple(i->(size(a, i) - 2g) ÷ 2 + 2g, Val(N)))
end


function simfiner(g, a::DeviceArray{T, N}) where {T, N}
    KA.zeros(get_backend(a), eltype(a), ntuple(i->(size(a, i) - 2g) * 2 + 2g, Val(N)))
end


function blocks(n, bsize)
    return div(n, bsize, RoundUp)
end


function blocks(g, a::AbstractArray, bsizes)
    blocks.(size(a) .- 2g, bsizes)
end


@generated function redblack(f, g, a::AbstractArray{T, N}, parity) where {T, N}
    if N == 2
        expr = quote
            redblack2(f, g, a, parity)
        end
    elseif N == 3
        expr = quote
            redblack3(f, g, a, parity)
        end
    else
        throw(ArgumentError("Only arrays with 2, 3 dimensions allowed"))
    end
    expr
end


function redblack2(f, g, a, parity)
    @kernel function kern(a)
        (i, j) = @index(Global, NTuple)
        
        i  = g + 2 * (i - 1) + 1
        j += g
        
        p = xor(parity, iseven(j - g))
        ind = CartesianIndex(i + p, j)
        
        f(ind)
    
        nothing
    end

    backend = get_backend(a)
    kern(backend)(a; ndrange=rbends(g, a))
    KA.synchronize(backend)

    nothing
end

function redblack3(f, g, a, parity)
    @kernel function kern(a)
        (i, j, k) = @index(Global, NTuple)
        
        i  = g + 2 * (i - 1) + 1
        j += g
        k += g

        # A -2g here does not change parity.
        p = xor(parity, iseven(k + j))
        ind = CartesianIndex(i + p, j, k)
        
        f(ind)
    
        nothing
    end

    backend = get_backend(a)
    kern(backend)(a; ndrange=rbends(g, a))
    KA.synchronize(backend)

    nothing
end



function residual!(g, r, u, b, s, c::AbstractConnector)
    st = lplstencil(u)

    @kernel function kern(r, @Const(u), @Const(b))
        ind = @index(Global, Cartesian)
        ind += oneunit(ind) * g
        
        @inline l = laplacian(g, u, ind, st, c)
        @inbounds r[ind] = s * b[ind] #+ l
        
        nothing
    end

    backend = get_backend(r)
    kernres2(backend)(r, u, b; ndrange=inends(g, r))
    KA.synchronize(backend)
    
    nothing
end


function restrict!(g, rh, r)
    st = cubestencil(r)
    
    @kernel function kern(rh, @Const(r))
        irh = @index(Global, Cartesian)  
        irh += oneunit(irh) * g
        
        ir = 2 * (irh - (g + 1) * oneunit(irh)) + (g + 1) * oneunit(irh)
        s = zero(eltype(r))
    
        for j in st
            @inbounds s += r[ir + j]
        end
        
        @inbounds rh[irh] = 4 * s / length(st)
    
        nothing
    end
    backend = get_backend(r)
    kern(backend)(rh, r; ndrange=inends(g, rh))
    KA.synchronize(backend)
    
    nothing
end


function interpolate!(g, r, rh, update::Type{Val{V}}=Val{false}) where {V}
    st = cubestencil(r)
    weights = binterpweights(st)
    @kernel function kern(r, @Const(rh))
        irh = @index(Global, Cartesian)  
        irh += oneunit(irh) * g

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

    backend = get_backend(r)
    kern(backend)(r, rh; ndrange=inends(g, rh))
    KA.synchronize(backend)

    nothing
end



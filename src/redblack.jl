function rbranges(g, a::AbstractArray{T, N}) where {T, N}
    t = ntuple(i -> (firstindex(a, i + 1) + g):(lastindex(a, i + 1) - g), Val(N - 1))
    h = (firstindex(a, 1) + g):2:(lastindex(a, 1) - g)

    (h, t...)
end


function rbends(g, a::AbstractArray{T, N}) where {T, N}
    t = ntuple(i -> lastindex(a, i + 1) - firstindex(a, i + 1) + 1 - 2g, Val(N - 1))

    l1 = lastindex(a, 1) - firstindex(a, 1) + 1 - 2g
    @assert iseven(l1)

    h = div(l1, 2)

    (h, t...)
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
        

function redblack2(f, g, a::AbstractArray{T, N}, parity) where {T, N}
    rng = rbranges(g, a)
    @batch for j in rng[2]
        p = xor(parity, iseven(j - g))
        for i in rng[1]
            f(CartesianIndex((i + p, j)))
        end
    end
end


function redblack3(f, g, a::AbstractArray{T, N}, parity) where {T, N}
    rng = rbranges(g, a)
            
    @batch for k in rng[3]   
        pk = xor(parity, iseven(k))
        for j in rng[2]
            pj = xor(pk, iseven(j))
            for i in rng[1]
                f(CartesianIndex((i + pj, j, k)))
            end
        end
    end
end

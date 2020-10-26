@generated function redblack(f, a::AbstractArray{T, N}, parity) where {T, N}
    if N == 2
        expr = quote
            redblack2(f, a, parity)
        end
    elseif N == 3
        expr = quote
            redblack3(f, a, parity)
        end
    else
        throw(ArgumentError("Only arrays with 2, 3 dimensions allowed"))
    end
    expr
end
        

function redblack2(f, a::AbstractArray{T, N}, parity) where {T, N}
    rng = rbranges(a)
    
    Threads.@threads for j in rng[2]
        p = xor(parity, iseven(j))
        for i in rng[1]
            f(CartesianIndex((i + p, j)))
        end
    end
end


function redblack3(f, a::AbstractArray{T, N}, parity) where {T, N}
    rng = rbranges(a)
            
    Threads.@threads for k in rng[3]   
        pk = xor(parity, iseven(k))
        for j in rng[2]
            pj = xor(pk, iseven(j))
            for i in rng[1]
                f(CartesianIndex((i + pj, j, k)))
            end
        end
    end
end


function rbranges(a::AbstractArray{T, N}) where {T, N}
    l = lastindex.(axes(a))
    t = ntuple(n -> 1:(l[n + 1] - 1), Val(N - 1))
    h = 1:2:l[1] - 1

    (h, t...)
end


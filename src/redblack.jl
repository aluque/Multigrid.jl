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



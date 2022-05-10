using LinearAlgebra
using JLD2, FileIO
using Printf
using Plots
using Plots.PlotMeasures
using LaTeXStrings
using Gnuplot

const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]
const σ2 = ComplexF64[0 -1im; 1im 0]
const σ3 = ComplexF64[1 0; 0 -1]

function construct_mesh(start::Vector{Float64}, goal::Vector{Float64}, nmesh::Int; endpoint=false)
    dr = (goal .- start) ./ nmesh
    num = ifelse(endpoint, nmesh+1, nmesh)

    [start .+ (n-1) .* dr for n in 1:num]
end

function calculate_Berry_phase(kpath::Vector{Vector{Float64}}, q::Vector{Float64}, param::Vector{Float64})
    nk::Int = 256

    # set wave number k
    ks::Vector{Vector{Float64}} = [Vector{Int}(undef, 3) for _ in 1:(nk*length(kpath))]
    for ik in 1:length(kpath)
        ks[(nk*(ik-1)+1):(nk*ik)] =
            construct_mesh(kpath[ik], circshift(kpath, -1)[ik], nk)
    end

    ψ::Array{ComplexF64, 3} = zeros(ComplexF64, (nk*length(kpath)), dim, Int(dim/2))
    γ::ComplexF64 = 1.0+0.0im
    # calculate ψ for all k1
    for i1 in 1:(nk*length(kpath))
        # define Hamiltonian and diagonalize it
        hamil = define_Hamiltonian_momentum(ks[i1], q, param)
        eigvec = eigvecs(hamil)

        # extract negative-energy eigenvectors
        ψ[i1, :, :] = eigvec[:, 1:Int(dim/2)]
    end

    # prepare shifted ψ
    ψ1 = circshift(ψ, (-1, 0, 0))

    # calculate Berry phase number using ψ
    for i1 in 1:(nk*length(kpath))
        U1 = det(ψ[i1, :, :]' * ψ1[i1, : ,:])
        γ *= U1 / abs(U1)
    end

    γ = log(γ) / (1.0im * π)

    real(γ)
end

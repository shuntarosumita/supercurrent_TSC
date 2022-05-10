module Rashba_2D_s

include("common_func.jl")
const dim = 4

function define_normal_part(k::Vector{Float64}, param::Vector{Float64})
    # set parameters
    kx, ky = k
    t1, t2, α1, α2, μ = param

    hamil_n::Matrix{ComplexF64} =
        (- μ - 2t1 * (cos(kx) + cos(ky)) - 4t2 * cos(kx) * cos(ky)) .* σ0 .+
        α1 .* (-sin(ky) .* σ1 .+ sin(kx) .* σ2) .+
        α2 * sin(kx) .* σ3

    hamil_n
end

function define_Hamiltonian_momentum(k::Vector{Float64}, q::Vector{Float64}, param::Vector{Float64})
    # set parameters
    t1, t2, α1, α2, μ, Δs = param

    # define Hamiltonian
    hamil_p::Matrix{ComplexF64} =
        define_normal_part(k .+ q, [t1, t2, α1, α2, μ])
    hamil_h::Matrix{ComplexF64} =
        - transpose(define_normal_part(-k .+ q, [t1, t2, α1, α2, μ]))
    order::Matrix{ComplexF64} = Δs .* (1im .* σ2)

    hamil::Matrix{ComplexF64} = vcat(
        hcat(hamil_p, order),
        hcat(order', hamil_h)
    )

    ishermitian(hamil) || error("Hamiltonian is not hermitian")

    hamil
end

function calculate_spectrum_pbc(q::Vector{Float64}, param::Vector{Float64})
    nmax = 1024

    # set wave number k
    k1s::Vector{Float64} = [π * (2 * i1 / nmax - 1) for i1 in 1:nmax]
    k2s::Vector{Float64} = [π * (2 * i2 / nmax - 1) for i2 in 1:nmax]

    eigval::Array{Float64, 3} = zeros(Float64, nmax, nmax, dim)
    for i2 in 1:nmax, i1 in 1:nmax
        # define Hamiltonian and diagonalize it
        hamil = define_Hamiltonian_momentum([k1s[i1], k2s[i2]], q, param)
        eigval[i1, i2, :] = eigvals(hamil)
    end

    # output data
    isdir("data") || mkdir("data")
    filename = @sprintf(
        "data/spec2Dp_s_%.3ft1_%.3ft2_%.3fa1_%.3fa2_%.3fm_%.3fds_%.3fqx_%.3fqy.jld2",
        param..., q...
    )
    datafile = File(format"JLD2", filename)
    save(datafile, "k1s", k1s, "k2s", k2s, "eigval", eigval)

    k1s, k2s, eigval
end

function plot_spectrum_pbc(q::Vector{Float64}, param::Vector{Float64})
    # get data of eigenvalues
    filename = @sprintf(
        "spec2Dp_s_%.3ft1_%.3ft2_%.3fa1_%.3fa2_%.3fm_%.3fds_%.3fqx_%.3fqy",
        param..., q...
    )
    isfile("data/" * filename * ".jld2") || error("no data exists")
    data = load("data/" * filename * ".jld2")
    eigval = data["eigval"]
    gap::Matrix{Float64} = eigval[:, :, Int(dim / 2) + 1] .- eigval[:, :, Int(dim / 2)]

    # gnuplot
    range = "[-pi*14/62:pi*14/62]"
    tics = "('−0.6' -0.6, '0.6' 0.6)"

    # plot eigenvalues
    @gsp data["k1s"] data["k2s"] gap "w pm3d notit" :-
    @gsp :- Gnuplot.palette(:roma) :-
    @gsp :- "set auto fix" "set size ratio -1" :-
    @gsp :- "set logscale cb" "set view map" :-
    @gsp :- "set xrange " * range "set yrange " * range :-

    # margins
    @gsp :- bma=0.16 tma=0.88 lma=0.05 rma=0.9 :-

    # labels
    @gsp :- "set label 1 '{/Helvetica:Italic E}' at graph 1.02, 1.06 left" :-
    @gsp :- "set xlabel '{/Helvetica:Italic k_x}' offset 0, 1.5" :-
    @gsp :- "set ylabel '{/Helvetica:Italic k_y}' rotate by 0 offset 2.5, 0" :-

    # ticks
    @gsp :- "set format cb '10^{%L}'" :-
    @gsp :- "set xtics " * tics * " scale 0.5" :-
    @gsp :- "set ytics " * tics * " scale 0.5" :-

    # output image file
    isdir("images") || mkdir("images")
    Gnuplot.save(
        term="pngcairo enhanced font 'Helvetica, 30' size 11in, 10in",
        output="images/"*filename*".png"
    )
end

# function plot_spectrum_pbc_old(q::Vector{Float64}, param::Vector{Float64})
#     # get data of eigenvalues
#     filename = @sprintf(
#         "spec2Dp_s_%.3ft1_%.3ft2_%.3fa1_%.3fa2_%.3fm_%.3fds_%.3fqx_%.3fqy",
#         param..., q...
#     )
#     isfile("data/" * filename * ".jld2") || error("no data exists")
#     data = load("data/" * filename * ".jld2")
#     eigval = data["eigval"]
#     gap::Matrix{Float64} = eigval[:, :, Int(dim / 2) + 1] .- eigval[:, :, Int(dim / 2)]

#     # plot eigenvalues
#     plt = heatmap(data["k1s"], data["k2s"], permutedims(log10.(gap[:, :]), (2, 1)),
#         xlims=(-π, π), ylims=(-π, π), c=cgrad(:roma, scale=:exp),
#         size=(480, 400), left_margin=0mm, right_margin=5mm, top_margin=3mm,
#         tickfont=font(14), guidefont=font(18),
#         xticks=([-π, π], ["−\\pi", "\\pi"]), yticks=([-π, π], ["−\\pi", "\\pi"]),
#         xlabel=L"k_x", ylabel=L"k_y", aspect_ratio=:equal, dpi=300
#     )

#     isdir("images") || mkdir("images")
#     savefig(plt, "images/" * filename * ".png")
#     plt
# end

function calculate_Berry_phase_mesh(q::Vector{Float64}, param::Vector{Float64})
    nmesh::Int = 31

    # calculate Berry phase on mesh
    dk::Float64 = 2π / nmesh
    ks::Vector{Float64} = [(i1 - nmesh/2) * dk for i1 in 1:nmesh]
    γmesh::Matrix{Float64} = [
        calculate_Berry_phase([[k1 - dk, k2 - dk], [k1, k2 - dk], [k1, k2], [k1 - dk, k2]], q, param)
        for k1 in ks, k2 in ks
    ]

    # output data
    isdir("data") || mkdir("data")
    filename = @sprintf(
        "data/berry2D_s_%.3ft1_%.3ft2_%.3fa1_%.3fa2_%.3fm_%.3fds_%.3fqx_%.3fqy.jld2",
        param..., q...
    )
    datafile = File(format"JLD2", filename)
    save(datafile, "ks", ks .- dk / 2, "γmesh", γmesh)

    ks .- dk / 2, γmesh
end

function plot_Berry_phase_mesh(q::Vector{Float64}, param::Vector{Float64})
    # get data of eigenvalues
    filename = @sprintf(
        "berry2D_s_%.3ft1_%.3ft2_%.3fa1_%.3fa2_%.3fm_%.3fds_%.3fqx_%.3fqy",
        param..., q...
    )
    isfile("data/" * filename * ".jld2") || error("no data exists")
    data = load("data/" * filename * ".jld2")
    display(filter(x -> abs(x) > 0.1, data["γmesh"]))

    # gnuplot
    range = "[-pi*14/62:pi*14/62]"
    tics = "('−0.6' -0.6, '0.6' 0.6)"

    # plot Berry phase
    @gp data["ks"] data["ks"] data["γmesh"] "w image notit" :-
    @gp :- "set cbrange [-1:1]" :-
    @gp :- Gnuplot.palette(:coolwarm) :-
    @gp :- "set auto fix" "set size ratio -1" :-
    @gp :- "set xrange " * range "set yrange " * range :-
    @gp :- "set x2range " * range "set y2range " * range :-

    # margins
    @gsp :- bma=0.16 tma=0.88 lma=0.15 rma=0.9 :-

    # labels
    @gp :- "set label 2 'γ/π' at graph 1.0, 1.06 left" :-
    @gp :- "set xlabel '{/Helvetica:Italic k_x}' offset 0, 1.0" :-
    @gp :- "set ylabel '{/Helvetica:Italic k_y}' rotate by 0 offset 4.0, 0" :-

    # ticks
    @gp :- "set xtics " * tics * " scale 0.5" :-
    @gp :- "set x2tics -pi, 2*pi/31, pi scale 0" :-
    @gp :- "set ytics " * tics * " scale 0.5" :-
    @gp :- "set y2tics -pi, 2*pi/31, pi scale 0" :-
    @gp :- "set format x2 ''" :-
    @gp :- "set format y2 ''" :-
    @gp :- "set grid front x2tics y2tics ls -1 lw 4 dt (4, 4)" :-

    # output image file
    isdir("images") || mkdir("images")
    Gnuplot.save(
        term="pngcairo enhanced font 'Helvetica, 30' size 11in, 10in",
        output="images/"*filename*".png"
    )
end

# function plot_Berry_phase_mesh_old(q::Vector{Float64}, param::Vector{Float64})
#     # get data of eigenvalues
#     filename = @sprintf(
#         "berry2D_s_%.3ft1_%.3ft2_%.3fa1_%.3fa2_%.3fm_%.3fds_%.3fqx_%.3fqy",
#         param..., q...
#     )
#     isfile("data/" * filename * ".jld2") || error("no data exists")
#     data = load("data/" * filename * ".jld2")
#     display(filter(x -> abs(x) > 0.1, data["γmesh"]))

#     # plot eigenvalues
#     plt = heatmap(data["ks"], data["ks"], permutedims(abs.(data["γmesh"]), (2, 1)),
#         xlims=(-π, π), ylims=(-π, π), clims=(0, 1), c=cgrad(:coolwarm),
#         size=(480, 400), left_margin=0mm, right_margin=5mm, top_margin=3mm,
#         tickfont=font(14), guidefont=font(18),
#         xticks=([-π, π], ["−\\pi", "\\pi"]), yticks=([-π, π], ["−\\pi", "\\pi"]),
#         xlabel=L"k_x", ylabel=L"k_y", aspect_ratio=:equal, dpi=300
#     )
#     nmesh::Int = length(data["ks"])
#     plot!(plt, [π * (2 * i1 / nmesh - 1) for i1 in 1:nmesh], seriestype=:hline,
#         linewidth=0.5, linestyle=:dot, linecolor=:black, label=:none
#     )
#     plot!(plt, [π * (2 * i1 / nmesh - 1) for i1 in 1:nmesh], seriestype=:vline,
#         linewidth=0.5, linestyle=:dot, linecolor=:black, label=:none
#     )

#     isdir("images") || mkdir("images")
#     savefig(plt, "images/" * filename * ".png")
#     plt
# end

end

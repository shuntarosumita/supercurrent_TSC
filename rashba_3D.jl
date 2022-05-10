module Rashba_3D

include("common_func.jl")
const dim = 4

function define_normal_part(k::Vector{Float64}, param::Vector{Float64})
    # set parameters
    kx, ky, kz = k
    t1, t2, tz, α1, α2, μ = param

    hamil_n::Matrix{ComplexF64} =
        (- μ - 2t1 * (cos(kx) + cos(ky)) - 4t2 * cos(kx) * cos(ky) - 2tz * cos(kz)) .* σ0 .+
        α1 .* (-sin(ky) .* σ1 .+ sin(kx) .* σ2) .+
        α2 * sin(kx) * sin(ky) * sin(kz) * (cos(kx) - cos(ky)) .* σ3

    hamil_n
end

function define_order_part(k::Vector{Float64}, param::Vector{Float64})
    # set parameters
    kx, ky, kz = k
    Δd, Δp = param

    # dx2-y2+p order parameter (B1)
    order::Matrix{ComplexF64} =
        (Δd * (cos(kx) - cos(ky)) .* σ0 .+
        Δp .* (sin(ky) .* σ1 .+ sin(kx) .* σ2)) * (1im .* σ2)

    # dxy+p order parameter (B2)
    # order::Matrix{ComplexF64} =
    #     (Δd * sin(kx) * sin(ky) .* σ0 .+
    #     Δp .* (sin(kx) .* σ1 .- sin(ky) .* σ2)) * (1im .* σ2)

    order
end

function define_Hamiltonian_momentum(k::Vector{Float64}, q::Vector{Float64}, param::Vector{Float64})
    # set parameters
    t1, t2, tz, α1, α2, μ, Δd, Δp = param

    # define Hamiltonian
    hamil_p::Matrix{ComplexF64} =
        define_normal_part(k .+ q, [t1, t2, tz, α1, α2, μ])
    hamil_h::Matrix{ComplexF64} =
        - transpose(define_normal_part(-k .+ q, [t1, t2, tz, α1, α2, μ]))
    order::Matrix{ComplexF64} = define_order_part(k, [Δd, Δp])

    hamil::Matrix{ComplexF64} = vcat(
        hcat(hamil_p, order),
        hcat(order', hamil_h)
    )

    ishermitian(hamil) || error("Hamiltonian is not hermitian")

    hamil
end

function define_Hamiltonian_real(direction::Int, kpara::Vector{Float64}, q::Vector{Float64}, param::Vector{Float64}, n3::Int; obc::Bool=true)
    if !(1 <= direction <= 3)
        @warn "direction=$direction is not applicable. We instead use direction=3."
        direction = 3
    end

    # set wave number k
    k3s::Vector{Float64} = [π * (2 * ik / n3 - 1) for ik in 1:n3]
    direction_others::Vector{Int} = deleteat!([1, 2, 3], direction)
    ks::Vector{Vector{Float64}} = [Vector{Int}(undef, 3) for _ in 1:n3]
    for ik in 1:n3
        ks[ik][direction] = k3s[ik]
        ks[ik][direction_others] = kpara
    end

    hamil::Matrix{ComplexF64} = zeros(ComplexF64, dim*n3, dim*n3)
    for j3 in 1:n3, i3 in 1:n3
        if obc && abs(i3 - j3) >= n3/2
            continue
        end

        # Fourier transformation
        hamil[(dim*(i3-1) + 1):(dim*i3), (dim*(j3-1) + 1):(dim*j3)] += sum(
            exp(1.0im * k3s[ik] * (i3 - j3)) / n3 .*
            define_Hamiltonian_momentum(ks[ik], q, param)
            for ik in 1:n3
        )
    end

    ishermitian(hamil) || error("Hamiltonian is not hermitian")

    Hermitian(hamil)
end

function define_Hamiltonian_obc_y(kpara::Vector{Float64}, qpara::Vector{Float64}, param::Vector{Float64}, ny::Int)
    # set parameters
    kx, kz = kpara
    kxp, kzp = kpara .+ qpara
    kxm, kzm = - kpara .+ qpara
    t1, t2, tz, α1, α2, μ, Δd, Δp = param

    # intra-unitcell terms
    intra = vcat(
        hcat(
            (- μ - 2t1 * cos(kxp) - 2tz * cos(kzp)) .* σ0 .+ α1 * sin(kxp) .* σ2,
            (Δd * cos(kx) .* σ0 .+ Δp * sin(kx) .* σ2) * (1im .* σ2)
        ),
        hcat(
            ((Δd * cos(kx) .* σ0 .+ Δp * sin(kx) .* σ2) * (1im .* σ2))',
            - transpose((- μ - 2t1 * cos(kxm) - 2tz * cos(kzm)) .* σ0 .+ α1 * sin(kxm) .* σ2)
        )
    )

    # inter-unitcell terms (nearest-neighbor)
    inter1 = [
        vcat(
            hcat(
                (- t1 - 2t2 * cos(kxp)) .* σ0 .+ (s / 2im) .* (- α1 .* σ1 .+ α2 * sin(kxp) * sin(kzp) * cos(kxp) .* σ3),
                (- (Δd / 2) .* σ0 .+ (s * Δp / 2im) .* σ1) * (1im .* σ2)
            ),
            hcat(
                ((- (Δd / 2) .* σ0 .- (s * Δp / 2im) .* σ1) * (1im .* σ2))',
                - transpose((- t1 - 2t2 * cos(kxm)) .* σ0 .- (s / 2im) .* (- α1 .* σ1 .+ α2 * sin(kxm) * sin(kzm) * cos(kxm) .* σ3))
            )
        )
        for s in [1, -1]
    ]

    # inter-unitcell terms (next-nearest-neighbor)
    inter2 = [
        vcat(
            hcat(
                - (s * α2 / 4im) * sin(kxp) * sin(kzp) .* σ3,
                zeros(ComplexF64, 2, 2)
            ),
            hcat(
                zeros(ComplexF64, 2, 2),
                - transpose((s * α2 / 4im) * sin(kxm) * sin(kzm) .* σ3)
            )
        )
        for s in [1, -1]
    ]

    # define Hamiltonian
    hamil::Matrix{ComplexF64} = zeros(ComplexF64, dim*ny, dim*ny)
    for iy in 1:ny
        # definition of index
        now = (dim*(iy-1) + 1):(dim*iy)
        ynn = (dim*iy + 1):(dim*(iy+1))
        ynnn = (dim*(iy+1) + 1):(dim*(iy+2))

        # intra-unitcell terms
        hamil[now, now] += intra

        # inter-unitcell terms (nearest-neighbor)
        if iy < ny
            hamil[now, ynn] += inter1[1]
            hamil[ynn, now] += inter1[2]
        end

        # inter-unitcell terms (next-nearest-neighbor)
        if iy < ny - 1
            hamil[now, ynnn] += inter2[1]
            hamil[ynnn, now] += inter2[2]
        end
    end

    ishermitian(hamil) || error("Hamiltonian is not hermitian")

    hamil
end

function define_Hamiltonian_obc_z(kpara::Vector{Float64}, qpara::Vector{Float64}, param::Vector{Float64}, nz::Int; pbc::Bool=false)
    # set parameters
    kx, ky = kpara
    kxp, kyp = kpara .+ qpara
    kxm, kym = - kpara .+ qpara
    t1, t2, tz, α1, α2, μ, Δd, Δp = param

    # intra-unitcell terms
    intra = vcat(
        hcat(
            (- μ - 2t1 * (cos(kxp) + cos(kyp)) - 4t2 * cos(kxp) * cos(kyp)) .* σ0 .+ α1 .* (sin(kxp) .* σ2 .- sin(kyp) .* σ1),
            (Δd * (cos(kx) - cos(ky)) .* σ0 .+ Δp .* (sin(kx) .* σ2 .+ sin(ky) .* σ1)) * (1im .* σ2)
        ),
        hcat(
            ((Δd * (cos(kx) - cos(ky)) .* σ0 .+ Δp .* (sin(kx) .* σ2 .+ sin(ky) .* σ1)) * (1im .* σ2))',
            - transpose((- μ - 2t1 * (cos(kxm) + cos(kym)) - 4t2 * cos(kxm) * cos(kym)) .* σ0 .+ α1 .* (sin(kxm) .* σ2 .- sin(kym) .* σ1))
        )
    )

    # inter-unitcell terms (nearest-neighbor)
    inter1 = [
       vcat(
            hcat(
                - tz .* σ0 .+ (s * α2 / 2im) * sin(kxp) * sin(kyp) * (cos(kxp) - cos(kyp)) .* σ3,
                zeros(ComplexF64, 2, 2)
            ),
            hcat(
                zeros(ComplexF64, 2, 2),
                - transpose(- tz .* σ0 .- (s * α2 / 2im) * sin(kxm) * sin(kym) * (cos(kxm) - cos(kym)) .* σ3)
            )
        )
        for s in [1, -1]
    ]

    # define Hamiltonian
    hamil::Matrix{ComplexF64} = zeros(ComplexF64, dim*nz, dim*nz)
    for iz in 1:nz
        # definition of index
        now = (dim*(iz-1) + 1):(dim*iz)
        znn = (dim*iz + 1):(dim*(iz+1))

        # intra-unitcell terms
        hamil[now, now] += intra

        # inter-unitcell terms (nearest-neighbor)
        if iz < nz
            hamil[now, znn] += inter1[1]
            hamil[znn, now] += inter1[2]
        end
    end

    # matrix elements for PBC
    if pbc
        hamil[(dim*(nz-1) + 1):(dim*nz), 1:dim] += inter1[1]
        hamil[1:dim, (dim*(nz-1) + 1):(dim*nz)] += inter1[2]
    end

    ishermitian(hamil) || error("Hamiltonian is not hermitian")

    hamil
end

function calculate_spectrum_k1k2(direction::Int, q::Vector{Float64}, param::Vector{Float64})
    nmax = 1024

    if !(1 <= direction <= 3)
        @warn "direction=$direction is not applicable. We instead use direction=3."
        direction = 3
    end
    direction_others::Vector{Int} = deleteat!([1, 2, 3], direction)

    # set wave number k
    k::Vector{Float64} = zeros(Float64, 3)
    k1s::Vector{Float64} = [π * i1 / nmax for i1 in 1:nmax]
    k2s::Vector{Float64} = [π * i2 / nmax for i2 in 1:nmax]

    eigval::Array{Float64, 4} = zeros(Float64, nmax, nmax, 2, dim)
    for i3 in 1:2
        k[direction] = π * (i3 - 1)
        for i2 in 1:nmax, i1 in 1:nmax
            k[direction_others[1]] = k1s[i1]
            k[direction_others[2]] = k2s[i2]

            # define Hamiltonian and diagonalize it
            hamil = define_Hamiltonian_momentum(k, q, param)
            eigval[i1, i2, i3, :] = eigvals(hamil)
        end
    end

    # output data
    isdir("data") || mkdir("data")
    filename = @sprintf(
        "data/spec3Dp_%d_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz.jld2",
        direction, param..., q...
    )
    datafile = File(format"JLD2", filename)
    FileIO.save(datafile, "k1s", k1s, "k2s", k2s, "eigval", eigval)

    k1s, k2s, eigval
end

function plot_spectrum_k1k2(direction::Int, q::Vector{Float64}, param::Vector{Float64}; ranges::Vector{String}=["[pi*13/32:pi*20/32]", "[pi*9/32:pi*16/32]"], tics::Vector{String}=["('1.3' 1.3, '1.9' 1.9)", "('0.9' 0.9, '1.5' 1.5)"])
    # get data of eigenvalues
    filename = @sprintf(
        "spec3Dp_%d_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz",
        direction, param..., q...
    )
    isfile("data/"*filename*".jld2") || error("no data exists")
    data = load("data/"*filename*".jld2")
    eigval = data["eigval"]
    gap::Array{Float64, 3} = eigval[:, :, :, Int(dim/2)+1] .- eigval[:, :, :, Int(dim/2)]

    # define labels
    direction_others::Vector{Int} = deleteat!([1, 2, 3], direction)
    title=[
        "'{/=40 (a) {/Helvetica:Italic k_z} = 0}'",
        "'{/=40 (b) {/Helvetica:Italic k_z} = π}'"
    ]
    cblabel = "'{/Helvetica:Italic E}'"
    xlabel = "'{/Helvetica:Italic k_$(["x", "y", "z"][direction_others[1]])}'"
    ylabel = "'{/Helvetica:Italic k_$(["x", "y", "z"][direction_others[2]])}'"

    # gnuplot
    @gsp "set multiplot layout 2, 1"
    for i3 in 1:2
        # plot eigenvalues
        @gsp :- i3 data["k1s"] data["k2s"] gap[:, :, i3] "w pm3d notit" :-
        @gsp :- Gnuplot.palette(:roma) :-
        @gsp :- "set auto fix" "set size ratio -1" :-
        @gsp :- "set logscale cb" "set view map" :-
        @gsp :- "set xrange " * ranges[i3] "set yrange " * ranges[i3] :-

        # margins
        @gsp :- bma=0.58-0.5*(i3-1) tma=0.94-0.5*(i3-1) :-
        @gsp :- lma=0.05 rma=0.9 :-

        # labels
        @gsp :- "set label 1 " * title[i3] * " at graph 0, 1.08 left" :-
        @gsp :- "set label 2 " * cblabel * " at graph 1.02, 1.06 left" :-
        @gsp :- "set xlabel " * xlabel * " offset 0, 1.5" :-
        @gsp :- "set ylabel " * ylabel * " rotate by 0 offset 2.5, 0" :-

        # ticks
        @gsp :- "set format cb '10^{%L}'" :-
        @gsp :- "set xtics " * tics[i3] * " scale 0.5" :-
        @gsp :- "set ytics " * tics[i3] * " scale 0.5" :-
    end

    # output image file
    isdir("images") || mkdir("images")
    Gnuplot.save(
        term="pngcairo enhanced font 'Helvetica, 30' size 11in, 20in",
        output="images/"*filename*".png"
    )
end

function plot_spectrum_k1k2_old(direction::Int, q::Vector{Float64}, param::Vector{Float64})
    # get data of eigenvalues
    filename = @sprintf(
        "spec3Dp_%d_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz",
        direction, param..., q...
    )
    isfile("data/"*filename*".jld2") || error("no data exists")
    data = load("data/"*filename*".jld2")
    eigval = data["eigval"]
    gap::Array{Float64, 3} = eigval[:, :, :, Int(dim/2)+1] .- eigval[:, :, :, Int(dim/2)]

    # plot eigenvalues
    direction_others::Vector{Int} = deleteat!([1, 2, 3], direction)
    title=[L"\mathrm{(a)}\ k_z = 0", L"\mathrm{(b)}\ k_z = \pi"]
    xlabel = latexstring("k_$(["x", "y", "z"][direction_others[1]])")
    ylabel = latexstring("k_$(["x", "y", "z"][direction_others[2]])")
    map = [
        heatmap(data["k1s"], data["k2s"],
            permutedims(log10.(gap[:, :, i3]), (2, 1)),
            xlims=(0, π), ylims=(0, π), c=cgrad(:roma, scale=:exp),
            title=title[i3], titleloc=:left, titlefont=font(18),
            tickfont=font(14), guidefont=font(18),
            xticks=([0, π], ["0", "\\pi"]), yticks=([0, π], ["0", "\\pi"]),
            xlabel=xlabel, ylabel=ylabel, aspect_ratio=:equal, dpi=300
        )
        for i3 in 1:2
    ]

    layout = @layout(grid(2, 1, height=[0.5, 0.5]))
    plt = plot(map[1], map[2], layout=layout,
        size=(480, 800), left_margin=0mm, right_margin=3mm, top_margin=0mm, bottom_margin=0mm
    )

    isdir("images") || mkdir("images")
    savefig(plt, "images/"*filename*".png")
    plt
end

function calculate_Berry_phase_mesh(q::Vector{Float64}, param::Vector{Float64})
    nmesh::Int = 32

    # calculate Berry phase on mesh
    dk::Float64 = π / nmesh
    ks::Vector{Float64} = [i1 * dk for i1 in 1:nmesh]
    γmesh::Vector{Matrix{Float64}} = [
        [
            calculate_Berry_phase([[k1-dk, k2-dk, k3], [k1, k2-dk, k3], [k1, k2, k3], [k1-dk, k2, k3]], q, param)
            for k1 in ks, k2 in ks
        ]
        for k3 in [0.0, Float64(π)]
    ]

    # output data
    isdir("data") || mkdir("data")
    filename = @sprintf(
        "data/berry3D_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz.jld2",
        param..., q...
    )
    datafile = File(format"JLD2", filename)
    FileIO.save(datafile, "ks", ks .- dk/2, "γmesh", γmesh)

    ks .- dk/2, γmesh
end

function plot_Berry_phase_mesh(q::Vector{Float64}, param::Vector{Float64}; ranges::Vector{String}=["[pi*13/32:pi*20/32]", "[pi*9/32:pi*16/32]"], tics::Vector{String}=["('1.3' 1.3, '1.9' 1.9)", "('0.9' 0.9, '1.5' 1.5)"])
    # get data of eigenvalues
    filename = @sprintf(
        "berry3D_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz",
        param..., q...
    )
    isfile("data/"*filename*".jld2") || error("no data exists")
    data = load("data/"*filename*".jld2")

    # define labels
    title=[
        "'{/=40 (c) {/Helvetica:Italic k_z} = 0}'",
        "'{/=40 (d) {/Helvetica:Italic k_z} = π}'"
    ]
    cblabel = "'γ/π'"
    xlabel = "'{/Helvetica:Italic k_x}'"
    ylabel = "'{/Helvetica:Italic k_y}'"

    # gnuplot
    @gp "set multiplot layout 2, 1"
    for i3 in 1:2
        # plot Berry phase
        @gp :- i3 data["ks"] data["ks"] data["γmesh"][i3] "w image notit" :-
        @gp :- "set cbrange [-1:1]" :-
        @gp :- Gnuplot.palette(:coolwarm) :-
        @gp :- "set auto fix" "set size ratio -1" :-
        @gp :- "set xrange " * ranges[i3] "set yrange " * ranges[i3] :-
        @gp :- "set x2range " * ranges[i3] "set y2range " * ranges[i3] :-

        # margins
        @gp :- bma=0.58-0.5*(i3-1) tma=0.94-0.5*(i3-1) :-
        @gp :- lma=0.15 rma=0.9 :-

        # labels
        @gp :- "set label 1 " * title[i3] * " at graph 0, 1.08 left" :-
        @gp :- "set label 2 " * cblabel * " at graph 1.0, 1.06 left" :-
        @gp :- "set xlabel " * xlabel * " offset 0, 1.0" :-
        @gp :- "set ylabel " * ylabel * " rotate by 0 offset 4.0, 0" :-

        # ticks
        @gp :- "set xtics " * tics[i3] * " scale 0.5" :-
        @gp :- "set x2tics 0, pi/32, pi scale 0" :-
        @gp :- "set ytics " * tics[i3] * " scale 0.5" :-
        @gp :- "set y2tics 0, pi/32, pi scale 0" :-
        @gp :- "set format x2 ''" :-
        @gp :- "set format y2 ''" :-
        @gp :- "set grid front x2tics y2tics ls -1 lw 4 dt (4, 4)" :-
    end

    # output image file
    isdir("images") || mkdir("images")
    Gnuplot.save(
        term="pngcairo enhanced font 'Helvetica, 30' size 11in, 20in",
        output="images/"*filename*".png"
    )
end

function plot_Berry_phase_mesh_old(q::Vector{Float64}, param::Vector{Float64})
    # get data of eigenvalues
    filename = @sprintf(
        "berry3D_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz",
        param..., q...
    )
    isfile("data/"*filename*".jld2") || error("no data exists")
    data = load("data/"*filename*".jld2")

    # plot eigenvalues
    title=[L"\mathrm{(a)}\ k_z = 0", L"\mathrm{(b)}\ k_z = \pi"]
    xlabel = latexstring("k_x")
    ylabel = latexstring("k_y")
    map = [
        heatmap(data["ks"], data["ks"], permutedims(data["γmesh"][i3], (2, 1)),
            xlims=(0, π), ylims=(0, π), c=cgrad(:coolwarm),
            title=title[i3], titleloc=:left, titlefont=font(18),
            tickfont=font(14), guidefont=font(18),
            xticks=([0, π], ["0", "\\pi"]), yticks=([0, π], ["0", "\\pi"]),
            xlabel=xlabel, ylabel=ylabel, aspect_ratio=:equal, dpi=300
        )
        for i3 in 1:2
    ]
    for i3 in 1:2
        plot!(map[i3], [i1 * π / 32 for i1 in 1:32], seriestype=:hline,
            linewidth=0.5, linestyle=:dot, linecolor=:black, label=:none
        )
        plot!(map[i3], [i1 * π / 32 for i1 in 1:32], seriestype=:vline,
            linewidth=0.5, linestyle=:dot, linecolor=:black, label=:none
        )
    end

    layout = @layout(grid(2, 1, height=[0.5, 0.5]))
    plt = plot(map[1], map[2], layout=layout,
        size=(480, 800), left_margin=0mm, right_margin=3mm, top_margin=0mm, bottom_margin=0mm
    )

    isdir("images") || mkdir("images")
    savefig(plt, "images/"*filename*".png")
    plt
end

function calculate_Chern_num(direction::Int, wave_num::Float64, q::Vector{Float64}, param::Vector{Float64})
    nmax::Int = 256

    # set wave number k
    k::Vector{Float64} = zeros(Float64, 3)
    k[direction] = wave_num
    direction_others::Vector{Int} = [[2, 3], [3, 1], [1, 2]][direction]

    # calculate ψ for all (k1, k2)
    ψ::Array{ComplexF64, 4} = zeros(ComplexF64, nmax, nmax, dim, Int(dim/2))
    for i2 in 1:nmax, i1 in 1:nmax
        k[direction_others[1]] = π * (2 * i1 / nmax - 1)
        k[direction_others[2]] = π * (2 * i2 / nmax - 1)

        # define Hamiltonian and diagonalize it
        hamil = define_Hamiltonian_momentum(k, q, param)
        eigvec = eigvecs(hamil)

        # extract negative-energy eigenvectors
        ψ[i1, i2, :, :] = eigvec[:, 1:Int(dim/2)]
    end

    # prepare shifted ψ
    ψ1 = circshift(ψ, (-1, 0, 0, 0))
    ψ2 = circshift(ψ, (0, -1, 0, 0))
    ψ12 = circshift(ψ, (-1, -1, 0, 0))

    # calculate Chern number using ψ
    F::Matrix{ComplexF64} = zeros(ComplexF64, nmax, nmax)
    for i2 in 1:nmax, i1 in 1:nmax
        # link variables
        U1_1 = det(ψ[i1, i2, :, :]' * ψ1[i1, i2, : ,:])
        U2_1 = det(ψ1[i1, i2, :, :]' * ψ12[i1, i2, : ,:])
        U1_2 = det(ψ2[i1, i2, :, :]' * ψ12[i1, i2, : ,:])
        U2_2 = det(ψ[i1, i2, :, :]' * ψ2[i1, i2, : ,:])

        # normalize the link variables
        U1_1 /= abs(U1_1)
        U2_1 /= abs(U2_1)
        U1_2 /= abs(U1_2)
        U2_2 /= abs(U2_2)

        # calculate field strength
        F[i1, i2] = log(U1_1 * U2_1 / (U2_2 * U1_2)) / (2.0im * π)
    end

    ν::ComplexF64 = sum(F)
    Fmax::Float64 = maximum(abs.(real.(F)))

    real(ν), Fmax
end

function calculate_Chern_num_kdep(direction::Int, n3::Int, q::Vector{Float64}, param::Vector{Float64})
    if !(1 <= direction <= 3)
        @warn "direction=$direction is not applicable. We instead use direction=3."
        direction = 3
    end

    # calculate k3 dependence
    k3s::Vector{Float64} = [π * (2 * i3 / n3 - 1) for i3 in 1:n3]
    νs::Vector{Float64} = Vector{Float64}(undef, n3)
    Fmaxs::Vector{Float64} = Vector{Float64}(undef, n3)
    for i3 in 1:n3
        νs[i3], Fmaxs[i3] = calculate_Chern_num(direction, k3s[i3], q, param)
    end

    νs, Fmaxs
end

function calculate_Chern_num_qin(qabs::Float64, param::Vector{Float64})
    n3::Int = 512
    ks::Vector{Float64} = [π * (2 * i3 / n3 - 1) for i3 in 1:n3]

    # calculate Chern numbers for in-plane q
    qin::Vector{Float64} = Vector{Float64}(undef, 3)
    νxs::Vector{Vector{Float64}} = [Vector{Float64}(undef, n3) for _ in 1:4]
    Fmaxxs::Vector{Vector{Float64}} = [Vector{Float64}(undef, n3) for _ in 1:4]
    νys::Vector{Vector{Float64}} = [Vector{Float64}(undef, n3) for _ in 1:4]
    Fmaxys::Vector{Vector{Float64}} = [Vector{Float64}(undef, n3) for _ in 1:4]
    for iφ in 1:4
        φ = (iφ-1)*π/6
        display(φ)
        qin = qabs .* [cos(φ), sin(φ), 0.0]
        νxs[iφ], Fmaxxs[iφ] = calculate_Chern_num_kdep(1, n3, qin, param)
        νys[iφ], Fmaxys[iφ] = calculate_Chern_num_kdep(2, n3, qin, param)
    end

    # output data
    isdir("data") || mkdir("data")
    filename = @sprintf(
        "data/chern3D_qin_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fq.jld2",
        param..., qabs
    )
    datafile = File(format"JLD2", filename)
    FileIO.save(
        datafile, "ks", ks,
        "νxs", νxs, "Fmaxxs", Fmaxxs, "νys", νys, "Fmaxys", Fmaxys
    )

    ks, νxs, Fmaxxs, νys, Fmaxys
end

function plot_Chern_num_qin(qabs::Float64, param::Vector{Float64})
    # get data of Chern numbers
    filename = @sprintf(
        "chern3D_qin_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fq",
        param..., qabs
    )
    isfile("data/"*filename*".jld2") || error("no data exists")
    data = load("data/"*filename*".jld2")
    νs = [data["νxs"], data["νys"]]
    println("x-axis Fmaxs = ", maximum.(data["Fmaxxs"]))
    println("y-axis Fmaxs = ", maximum.(data["Fmaxys"]))

    # define labels
    title=["'{/=40 (a)}'", "'{/=40 (b)}'"]
    xlabel = ["'{/Helvetica:Italic k_x}'",  "'{/Helvetica:Italic k_y}'"]
    ylabel = ["'ν({/Helvetica:Italic k_x})'", "'ν({/Helvetica:Italic k_y})'"]

    # gnuplot
    dts = " dt " .* ["'-'", "'.-'", "'_'", "(1, 0)"]
    colors = " lc rgb " .* ["'red'", "'dark-green'", "'blue'", "'orange'"]
    @gp "set multiplot layout 2, 1"
    for idir in 1:2, iφ in 1:4
        rat = (iφ-1) // 6
        label = " title 'φ = " * ifelse(
            rat == 0,
            "0",
            replace(replace(string(rat), "1//" => "π/"), "//" => "π/")
        ) * "'"

        # plot Chern numbers
        @gp :- idir data["ks"] νs[idir][iφ] "w l lw $(18-4*iφ) " * dts[iφ] * colors[iφ] * label :- # ps $(4.5-iφ)
        @gp :- "set key right " * ["bottom", "top"][idir] :-
        @gp :- xr=[0, π] yr=[[-5, 1], [-1, 5]][idir] :-

        # margins
        # @gp :- bma=0.62-0.5*(idir-1) tma=0.94-0.5*(idir-1) :-
        # @gp :- lma=0.15 rma=0.95 :-
        @gp :- bma=0.24 tma=0.88 :-
        @gp :- lma=0.075+0.5*(idir-1) rma=0.475+0.5*(idir-1) :-

        # labels
        @gp :- "set label 1 " * title[idir] * " at graph -0.05, 1.1 right" :-
        @gp :- "set xlabel " * xlabel[idir] :-
        @gp :- "set ylabel " * ylabel[idir] :-

        # ticks
        @gp :- "set xtics ('−π' -pi, '−π/2' -pi/2, '0' 0.0, 'π/2' pi/2, 'π' pi) scale 0.5" :-
        @gp :- "set ytics ('−4' -4.0, '−2' -2.0, '0' 0.0, '2' 2.0, '4' 4.0) scale 0.5" :-
        @gp :- "set grid front xtics ytics lw 1 lt -1 dt 3" :-
    end

    # output image file
    isdir("images") || mkdir("images")
    Gnuplot.save(
        term="pngcairo enhanced font 'Helvetica, 30' size 24in, 8in",
        output="images/"*filename*".png"
    )
end

# function plot_Chern_num_kdep_old(direction::Int, q::Vector{Float64}, param::Vector{Float64})
#     # get data of Chern numbers
#     filename = @sprintf(
#         "chern3D_%d_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz",
#         direction, param..., q...
#     )
#     isfile("data/"*filename*".jld2") || error("no data exists")
#     data = load("data/"*filename*".jld2")
#     println(maximum(data["Fmaxs"]))

#     # plot Chern numbers
#     xlabel = latexstring("k_$(["x", "y", "z"][direction])")
#     plt = plot([0], seriestype=:hline,
#         linewidth=1, linestyle=:dash, linecolor=:black,
#         xlims=(-π, π), ylims=(-5, 1),
#         tickfont=font(14), guidefont=font(18),
#         xticks=([-π:π/2:π;], ["−π", "−π/2", "0", "π/2", "π"]),
#         yticks=([-4:2:0;], ["−4", "−2", "0"]),
#         label=:none, xlabel=xlabel, ylabel=L"\nu",
#         size=(400, 240), dpi=300,
#         left_margin=2mm, right_margin=1mm, top_margin=0mm, bottom_margin=3mm
#     )
#     plot!(plt, data["ks"], data["νs"], linecolor=:blue, label=:none)

#     isdir("images") || mkdir("images")
#     savefig(plt, "images/"*filename*".png")
#     plt
# end

function calculate_spectrum_obc(direction::Int, k2::Float64, q::Vector{Float64}, param::Vector{Float64})
    nmax = 512
    n3 = 256
    if direction != 1 && direction != 3
        error("incorrect direction")
    end

    # calculate energy eigenvalues for all k1
    k1s::Vector{Float64} = [π * (2 * i1 / nmax - 1) for i1 in 1:nmax]
    spec_o::Array{Float64} = zeros(Float64, nmax, dim*n3)
    spec_p::Array{Float64} = zeros(Float64, nmax, dim*n3)
    for i1 in 1:nmax
        kpara = ifelse(direction == 1, [k2, k1s[i1]], [k1s[i1], k2])
        hamil = define_Hamiltonian_real(direction, kpara, q, param, n3, obc=true)
        spec_o[i1, :] = eigvals(hamil)
        hamil = define_Hamiltonian_real(direction, kpara, q, param, n3, obc=false)
        spec_p[i1, :] = eigvals(hamil)
    end

    # output data
    isdir("data") || mkdir("data")
    filename = @sprintf(
        "data/spec3Do_%d_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz_%.3fk2.jld2",
        direction, param..., q..., k2
    )
    datafile = File(format"JLD2", filename)
    FileIO.save(datafile, "k1s", k1s, "spec_o", spec_o, "spec_p", spec_p)

    k1s, spec_o, spec_p
end

function plot_spectrum_obc(direction::Int, k2::Float64, q::Vector{Float64}, param::Vector{Float64}, inset_xr::Vector{Float64}, inset_yr::Vector{Float64})
    # get data of spectrum
    filename = @sprintf(
        "spec3Do_%d_%.3ft1_%.3ft2_%.3ftz_%.3fa1_%.3fa2_%.3fm_%.3fdd_%.3fdp_%.3fqx_%.3fqy_%.3fqz_%.3fk2",
        direction, param..., q..., k2
    )
    isfile("data/"*filename*".jld2") || error("no data exists")
    data = load("data/"*filename*".jld2")
    k1s = data["k1s"]
    spec_o = data["spec_o"]
    spec_p = data["spec_p"]

    # define labels
    xlabel = ifelse(
        direction == 1,
        "'{/Helvetica:Italic k_z}'",
        "'{/Helvetica:Italic k_x}'"
    )

    # gnuplot
    @gp "set multiplot"

    # main margins
    @gp :- 1 bma=0.22 tma=0.95 lma=0.12 rma=0.95 :-

    # main labels
    @gp :- "set xlabel " * xlabel :-
    @gp :- "set ylabel '{/Helvetica:Italic E}' rotate by 0 offset 1.0, 0" :-

    # main ticks
    @gp :- xr=[-π, π] yr=[-1.0, 1.0] :-
    @gp :- "set xtics ('−π' -pi, '0' 0.0, 'π' pi) scale 0.5" :-
    @gp :- "set ytics ('−1' -1.0, '0' 0.0, '1' 1.0) scale 0.5" :-
    @gp :- "set grid front xtics ytics lw 1 lt -1 dt 3" :-

    # main plot eigenvalues
    for is in 1:size(spec_o)[2]
        @gp :- k1s spec_o[:, is] "w l lw 2 dt '_' lc rgb 'red' notitle" :-
        @gp :- k1s spec_p[:, is] "w l lw 2 lc rgb 'blue' notitle" :-
    end

    # main rectangle
    @gp :- "set object 1 rectangle from first $(inset_xr[1]), $(inset_yr[1]) to first $(inset_xr[2]), $(inset_yr[2]) front fs empty border lc rgb 'orange' lw 4 dt (15, 5, 5, 5)" :-

    # inset margins
    @gp :- 2 bma=0.25 tma=0.7 lma=0.57 rma=0.92 :-
    @gp :- "set object 1 rectangle from graph 0, 0 to graph 1, 1 behind fs solid border lc rgb 'black' lw 1 fc rgb 'white'" :-

    # inset labels and ticks
    @gp :- "unset xlabel" "unset ylabel" :-
    @gp :- xr=inset_xr yr=inset_yr :-
    @gp :- "unset grid" :-

    # inset plot eigenvalues
    for is in 1:size(spec_o)[2]
        @gp :- k1s spec_o[:, is] "w l lw 4 dt '_' lc rgb 'red' notitle" :-
        @gp :- k1s spec_p[:, is] "w l lw 2 lc rgb 'blue' notitle" :-
    end

    # output image file
    isdir("images") || mkdir("images")
    Gnuplot.save(
        term="pngcairo enhanced font 'Helvetica, 30' size 15in, 8in",
        output="images/"*filename*".png"
    )
end

end
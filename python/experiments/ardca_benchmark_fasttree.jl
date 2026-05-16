#!/usr/bin/env julia
# ArDCA ancestral reconstruction benchmark on Pfam val families
# using FastTree ML trees instead of Neighbor-Joining trees.
#
# Held-out leaf prediction: hold out one leaf, fit ArDCA on the rest,
# reconstruct ancestor using a pre-computed FastTree tree, score
# per-column accuracy vs held-out leaf.
#
# Usage:
#   export PATH="$HOME/.juliaup/bin:$PATH"
#   julia experiments/ardca_benchmark_fasttree.jl

using ArDCA
using AncestralSequenceReconstruction
using TreeTools
using JSON
using Random
using LinearAlgebra
using Statistics
using Printf

const ASR = AncestralSequenceReconstruction

# ── Alphabet mapping ──────────────────────────────────────────────────
# ArDCA (DCAUtils): ACDEFGHIKLMNPQRSTVWY- = 1..20 gap=21
# ASR:              -ACDEFGHIKLMNPQRSTVWY = gap=1, A=2..Y=21

ardca_to_asr(x::Integer) = x == 21 ? 1 : x + 1
asr_to_ardca(x::Integer) = x == 1 ? 21 : x - 1

const ARDCA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
const ARDCA_CHARS = "ACDEFGHIKLMNPQRSTVWY-"

function char_to_ardca(c::Char)
    idx = findfirst(==(uppercase(c)), ARDCA_ALPHABET)
    return isnothing(idx) ? 21 : idx  # gap, '.', and unknown -> 21
end

ardca_int_to_char(x::Integer) = ARDCA_CHARS[x]

# ── Stockholm parser ─────────────────────────────────────────────────
"""
Parse a Stockholm-format MSA file.
Returns (seqs::Dict{String,String}, names::Vector{String}).
Sequences are raw aligned strings (uppercase + '-' + '.' etc).
"""
function parse_stockholm(filepath::String)
    seqs = Dict{String,String}()
    order = String[]
    open(filepath) do f
        for line in eachline(f)
            stripped = strip(line)
            (isempty(stripped) || startswith(stripped, "#") ||
             startswith(stripped, "//")) && continue
            parts = split(stripped)
            length(parts) >= 2 || continue
            name = String(parts[1])
            seq = String(parts[2])
            if haskey(seqs, name)
                seqs[name] *= seq
            else
                seqs[name] = seq
                push!(order, name)
            end
        end
    end
    return seqs, order
end

# ── MSA to integer matrix (ArDCA encoding) ───────────────────────────
"""
Convert aligned sequences to ArDCA integer matrix (N_sites x N_seqs).
Returns: Z matrix, sequence names.
"""
function msa_to_ardca_matrix(seqs::Dict{String,String}, names::Vector{String})
    L = length(seqs[names[1]])
    M = length(names)
    Z = Matrix{Int8}(undef, L, M)
    for (j, name) in enumerate(names)
        seq = seqs[name]
        @assert length(seq) == L "Sequence $name has length $(length(seq)) != $L"
        for i in 1:L
            Z[i, j] = char_to_ardca(seq[i])
        end
    end
    return Z
end

# ── Write FASTA for ArDCA ────────────────────────────────────────────
function write_fasta(filepath::String, names::Vector{String}, Z::Matrix{<:Integer})
    open(filepath, "w") do f
        N, M = size(Z)
        for j in 1:M
            println(f, ">$(names[j])")
            println(f, String([ardca_int_to_char(Z[i, j]) for i in 1:N]))
        end
    end
end

# ── Neighbor-Joining ─────────────────────────────────────────────────
"""
Pairwise JC-corrected distance matrix from ArDCA-encoded alignment.
"""
function pairwise_distances(Z::Matrix{<:Integer})
    N, M = size(Z)
    D = zeros(Float64, M, M)
    for i in 1:M, j in (i+1):M
        ndiff = 0; nvalid = 0
        for k in 1:N
            if Z[k, i] != 21 && Z[k, j] != 21
                nvalid += 1
                ndiff += (Z[k, i] != Z[k, j])
            end
        end
        d = nvalid > 0 ? ndiff / nvalid : 1.0
        d = d < 20/21 ? -20/21 * log(1 - 21/20 * d) : 5.0
        D[i, j] = D[j, i] = d
    end
    return D
end

"""
Sanitize a label for Newick format: replace problematic chars.
"""
function sanitize_newick_label(s::String)
    # Replace characters that break Newick parsing
    s = replace(s, "/" => "_SLASH_")
    s = replace(s, "(" => "_LP_")
    s = replace(s, ")" => "_RP_")
    s = replace(s, ":" => "_COL_")
    s = replace(s, "," => "_COM_")
    s = replace(s, ";" => "_SC_")
    s = replace(s, " " => "_SP_")
    s = replace(s, "'" => "_Q_")
    s = replace(s, "[" => "_LB_")
    s = replace(s, "]" => "_RB_")
    return s
end

"""
Neighbor-Joining tree construction. Returns Newick string.
Names are sanitized for Newick compatibility.
"""
function neighbor_joining(D::Matrix{Float64}, names::Vector{String})
    n = size(D, 1)
    n >= 3 || error("Need >= 3 taxa")

    active = collect(1:n)
    dist = copy(D)
    labels = [sanitize_newick_label(n) for n in names]
    next_id = 1

    while length(active) > 2
        m = length(active)
        r = zeros(m)
        for i in 1:m, j in 1:m
            i != j && (r[i] += dist[active[i], active[j]])
        end

        best_q, best_i, best_j = Inf, 1, 2
        for i in 1:m, j in (i+1):m
            q = (m - 2) * dist[active[i], active[j]] - r[i] - r[j]
            q < best_q && (best_q = q; best_i = i; best_j = j)
        end

        ai, aj = active[best_i], active[best_j]
        dij = dist[ai, aj]
        bl_i = max(dij / 2 + (r[best_i] - r[best_j]) / (2 * (m - 2)), 1e-6)
        bl_j = max(dij - bl_i, 1e-6)

        new_label = "NJ_$(next_id)"
        next_id += 1

        # Expand distance matrix
        old_sz = size(dist, 1)
        new_dist = zeros(old_sz + 1, old_sz + 1)
        new_dist[1:old_sz, 1:old_sz] .= dist
        for k_idx in 1:m
            (k_idx == best_i || k_idx == best_j) && continue
            ak = active[k_idx]
            d_new = max((dist[ai, ak] + dist[aj, ak] - dij) / 2, 1e-6)
            new_dist[old_sz+1, ak] = d_new
            new_dist[ak, old_sz+1] = d_new
        end
        dist = new_dist

        subtree = "($(labels[ai]):$(@sprintf("%.6f",bl_i)),$(labels[aj]):$(@sprintf("%.6f",bl_j)))$new_label"
        push!(labels, subtree)  # labels[old_sz + 1] = subtree
        push!(active, old_sz + 1)
        deleteat!(active, sort([best_i, best_j]))
    end

    ai, aj = active[1], active[2]
    d = max(dist[ai, aj], 1e-6) / 2
    return "($(labels[ai]):$(@sprintf("%.6f",d)),$(labels[aj]):$(@sprintf("%.6f",d)));"
end

# ── Main benchmark ───────────────────────────────────────────────────
function run_benchmark(;
    unified_json = joinpath(@__DIR__, "unified_reconstruction_benchmark.json"),
    pfam_dir     = expanduser("~/bio-datasets/data/pfam/seed"),
    tree_dir     = expanduser("~/bio-datasets/data/pfam-seed/trees"),
    output_file  = joinpath(@__DIR__, "ardca_benchmark_fasttree_results.json"),
)
    # Read the exact family list and held-out leaves from the unified
    # benchmark, so results are directly comparable.
    unified = JSON.parsefile(unified_json)
    unified_results = unified["results"]
    println("Loaded $(length(unified_results)) families from unified benchmark")

    results = Dict{String,Any}[]
    done_families = Set{String}()
    # Resume: load existing results if output file exists
    if isfile(output_file)
        try
            existing = JSON.parsefile(output_file)
            if haskey(existing, "families")
                for r in existing["families"]
                    push!(results, r)
                    push!(done_families, r["family"])
                end
                println("Resume: loaded $(length(done_families)) existing results")
            end
        catch e
            @warn "Could not load existing results: $e"
        end
    end
    accuracies = Float64[]
    n_processed = 0
    n_skipped = 0
    n_errors = 0

    for (idx, ures) in enumerate(unified_results)
        family = ures["family"]
        family in done_families && continue
        unified_holdout = ures["held_out"]

        sto_file = joinpath(pfam_dir, "$(family).sto")
        isfile(sto_file) || continue

        # Parse Stockholm
        seqs, seq_names = try
            parse_stockholm(sto_file)
        catch e
            @warn "Parse failed for $family: $e"
            n_errors += 1; continue
        end

        nseqs = length(seq_names)
        nseqs < 4 && (n_skipped += 1; continue)

        # Convert to matrix
        Z_full = try
            msa_to_ardca_matrix(seqs, seq_names)
        catch e
            @warn "Matrix conversion failed for $family: $e"
            n_errors += 1; continue
        end
        N_sites = size(Z_full, 1)

        println("\n[$idx/$(length(unified_results))] $family: $nseqs seqs, $N_sites cols")

        try
            # Use the SAME held-out leaf as the unified benchmark
            holdout_name = unified_holdout
            holdout_idx = findfirst(==(holdout_name), seq_names)
            if isnothing(holdout_idx)
                println("  Skip: holdout '$holdout_name' not in MSA")
                n_skipped += 1; continue
            end
            holdout_seq = Z_full[:, holdout_idx]

            remain_idx = setdiff(1:nseqs, holdout_idx)
            remain_names = seq_names[remain_idx]
            Z_remain = Z_full[:, remain_idx]

            length(remain_names) < 3 && (n_skipped += 1; println("  Skip: <3 remaining"); continue)

            # Fit ArDCA using the matrix API directly.
            # ArDCA infers q = maximum(Z), so if the MSA has no gaps
            # (max=20), q<21 and ASR fails. Fix: clamp max to 21 by
            # setting Z[1,1] = max(Z[1,1], 21) — this is a no-op when
            # gaps are already present, and forces q=21 otherwise
            # without adding fake positions or sequences.
            Z_fit = copy(Z_remain)
            if maximum(Z_fit) < 21
                Z_fit[1, 1] = Int8(21)  # inject one gap symbol
            end
            W = fill(1.0 / size(Z_fit, 2), size(Z_fit, 2))  # uniform weights

            println("  Fitting ArDCA...")
            arnet, arvar = ardca(Z_fit, W; verbose=false)

            # Load pre-computed FastTree ML tree.
            # FastTree trees use the original MSA names (with / etc),
            # so we do NOT sanitize names here — use them as-is.
            tree_file = joinpath(tree_dir, "$(family).nwk")
            if !isfile(tree_file)
                println("  Skip: no FastTree at $tree_file")
            end
            println("  Loading FastTree ML tree...")
            newick_str = strip(read(tree_file, String))
            tree = parse_newick_string(newick_str)

            # Use original names (no sanitization) for tree lookups
            san_names = seq_names
            san_holdout = holdout_name
            san_remain = remain_names
            san_to_idx = Dict(san_names[i] => i for i in 1:nseqs)

            # Record holdout's parent and branch length BEFORE pruning
            holdout_node = tree[san_holdout]
            parent_node = ancestor(holdout_node)
            parent_lbl = label(parent_node)
            holdout_bl = branch_length(holdout_node)

            # Prune holdout from tree
            println("  Pruning holdout and running ASR...")
            prune!(tree, san_holdout)

            # Build AutoRegressiveModel
            q = length(arnet.p0)
            if q != 21
                @warn "ArDCA q=$q (expected 21); dummy gap sequence may not have helped"
                n_errors += 1
                continue
            end
            ar_model = AutoRegressiveModel(arnet)

            # Prepare leaf sequences (ASR encoding)
            leaf_seqs = Pair{String,String}[]
            for (i, sname) in enumerate(san_remain)
                j = findfirst(==(remain_names[i]), seq_names)
                seq_ints = Z_full[:, j]
                asr_str = String([ASR._AA_ALPHABET[ardca_to_asr(x)] for x in seq_ints])
                push!(leaf_seqs, sname => asr_str)
            end

            # Run ASR
            strategy = ASRMethod(; ML=true, optimize_branch_length=false)
            asr_tree, internal_seqs = infer_ancestral(tree, leaf_seqs, ar_model, strategy)

            # Find reconstruction at the closest surviving ancestor
            # Try parent first, then any available internal node
            recon_label = nothing
            recon_str = nothing

            if haskey(internal_seqs, parent_lbl)
                recon_label = parent_lbl
                recon_str = internal_seqs[parent_lbl]
            else
                # Parent was collapsed. Use root (closest surviving ancestor).
                root_lbl = label(root(asr_tree))
                if haskey(internal_seqs, root_lbl)
                    recon_label = root_lbl
                    recon_str = internal_seqs[root_lbl]
                else
                    # Fallback: use first available internal sequence
                    for (k, v) in internal_seqs
                        recon_label = k
                        recon_str = v
                        break
                    end
                end
            end

            if isnothing(recon_str)
                @warn "No reconstruction found for $family"
                n_errors += 1
                continue
            end

            # Convert reconstruction to ArDCA encoding for comparison
            recon_ints = [asr_to_ardca(ASR.aa_alphabet.mapping[c]) for c in recon_str]

            # Per-column accuracy (including gaps)
            n_correct = sum(recon_ints[i] == holdout_seq[i] for i in 1:N_sites)
            accuracy = n_correct / N_sites

            push!(accuracies, accuracy)
            n_processed += 1

            # Extract predicted sequence: strip gaps, convert to 0-indexed ints
            # ArDCA encoding: 1-20 = AA, 21 = gap → 0-indexed: subtract 1, skip gaps
            pred_seq_0idx = [x - 1 for x in recon_ints if x != 21]
            # Also extract true held-out sequence (ungapped, 0-indexed)
            true_seq_0idx = [x - 1 for x in holdout_seq if x != 21]

            result = Dict(
                "family" => family,
                "n_seqs" => nseqs,
                "n_cols" => N_sites,
                "held_out" => holdout_name,
                "recon_node" => recon_label,
                "holdout_branch_length" => holdout_bl,
                "accuracy" => accuracy,
                "n_correct" => n_correct,
                "pred_seq" => pred_seq_0idx,
                "pred_len" => length(pred_seq_0idx),
                "true_seq" => true_seq_0idx,
                "true_len" => length(true_seq_0idx),
            )
            push!(results, result)

            @printf("  Accuracy: %.1f%% (%d/%d) [recon at %s]\n",
                    100*accuracy, n_correct, N_sites, recon_label)


            # Save intermediate results every 10 families
            if n_processed % 10 == 0
                _save_results(output_file, results, accuracies,
                              n_processed, n_skipped, n_errors)
                println("  [saved intermediate results]")
            end

        catch e
            @warn "Error on $family" exception=(e, catch_backtrace())
            n_errors += 1
            continue
        end
    end

    # Final save
    _save_results(output_file, results, accuracies, n_processed, n_skipped, n_errors)

    # Summary
    println("\n" * "="^60)
    println("ArDCA Ancestral Reconstruction Benchmark")
    println("="^60)
    @printf("Families processed: %d\n", n_processed)
    @printf("Families skipped:   %d\n", n_skipped)
    @printf("Families errored:   %d\n", n_errors)
    if !isempty(accuracies)
        @printf("Mean accuracy:      %.1f%%\n", 100*mean(accuracies))
        @printf("Median accuracy:    %.1f%%\n", 100*median(accuracies))
        @printf("Std accuracy:       %.1f%%\n", 100*std(accuracies))
    end
    println("="^60)
end

function _save_results(output_file, results, accuracies, n_proc, n_skip, n_err)
    summary = Dict(
        "method" => "ArDCA",
        "n_families" => n_proc,
        "n_skipped" => n_skip,
        "n_errors" => n_err,
        "mean_accuracy" => isempty(accuracies) ? NaN : mean(accuracies),
        "median_accuracy" => isempty(accuracies) ? NaN : median(accuracies),
        "std_accuracy" => isempty(accuracies) ? NaN : std(accuracies),
        "min_accuracy" => isempty(accuracies) ? NaN : minimum(accuracies),
        "max_accuracy" => isempty(accuracies) ? NaN : maximum(accuracies),
        "families" => results,
        "comparison" => Dict(
            "felsenstein_LG08" => 0.543,
            "felsenstein_C10" => 0.543,
            "felsenstein_C20" => 0.543,
            "carabs" => 0.713,
        ),
    )
    open(output_file, "w") do f
        JSON.print(f, summary, 2)
    end
end

# Run when executed directly (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end

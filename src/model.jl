using Lux: AbstractLuxLayer, BatchNorm, Dense, Dropout, logsoftmax, sigmoid, softplus
using Random: AbstractRNG, default_rng, seed!
using Statistics: mean

import Lux: initialparameters, initialstates

export ConvLayer, Model

"""
    cgcnn_pooling(atom_fea, crystal_atom_idx)
atom_fea :: (N, F)
crystal_atom_idx :: Vector of index vectors, length N0
Returns:
    (N0, F) matrix of per-crystal mean pooled features.
"""
function cgcnn_pooling(atom_fea, crystal_atom_idx)
    @assert sum(length.(crystal_atom_idx)) == size(atom_fea, 1)
    pooled = map(crystal_atom_idx) do idxs
        mean(atom_fea[idxs, :]; dims=1)
    end
    return vcat(pooled...)
end

struct ConvLayer{A,B,C} <: AbstractLuxLayer
    atom_feat_len::UInt32
    neighbor_feat_len::UInt32
    dense::A
    bn1::B
    bn2::C
end
function ConvLayer(atom_feat_len, neighbor_feat_len)
    dense = Dense((2atom_feat_len + neighbor_feat_len) => (2atom_feat_len))
    bn1 = BatchNorm(2atom_feat_len)
    bn2 = BatchNorm(atom_feat_len)
    return ConvLayer{typeof(dense),typeof(bn1),typeof(bn2)}(
        atom_feat_len, neighbor_feat_len, dense, bn1, bn2
    )
end

function initialparameters(rng::AbstractRNG, l::ConvLayer)
    return (
        dense=initialparameters(rng, l.dense),
        bn1=initialparameters(rng, l.bn1),
        bn2=initialparameters(rng, l.bn2),
    )
end

function initialstates(rng::AbstractRNG, l::ConvLayer)
    # Dense has no state; BN does
    return (bn1=initialstates(rng, l.bn1), bn2=initialstates(rng, l.bn2))
end

"""
Forward pass for CGConvLayer.
Inputs:
- atom_in_fea: (N, F)
- nbr_fea:     (N, M, B)
- nbr_fea_idx: (N, M)  -- must be 1-based indices in Julia
Returns:
- atom_out_fea: (N, F)
"""
function (layer::ConvLayer)((atom_in_fea, nbr_fea, nbr_fea_idx), ps, st)
    F, B, (N, M) = layer.atom_feat_len, layer.neighbor_feat_len, size(atom_in_fea)
    # Gather neighbor atom features: (N, M, F)
    atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
    # Expand center atom features to (N, M, F)
    center = repeat(reshape(atom_in_fea, N, 1, F), 1, M, 1)
    # Concatenate: (N, M, 2F+B)
    total_nbr_fea = cat(center, atom_nbr_fea, nbr_fea; dims=3)
    # Apply fc_full on last dim via reshape/permutation
    # total_nbr_fea: (N, M, 2F+B) -> X: (2F+B, N*M)
    X = reshape(permutedims(total_nbr_fea, (3, 1, 2)), 2F + B, :)
    Y, _ = layer.dense(X, ps.dense, NamedTuple())  # Dense is stateless
    # Back to (N, M, 2F)
    total_gated_fea = permutedims(reshape(Y, 2F, N, M), (2, 3, 1))
    # BN1 over (N*M, 2F)
    Xbn1 = reshape(permutedims(total_gated_fea, (3, 1, 2)), 2F, :)
    Ybn1, st_bn1 = layer.bn1(Xbn1, ps.bn1, st.bn1)
    total_gated_fea = permutedims(reshape(Ybn1, 2F, N, M), (2, 3, 1))
    # Split filter/core: each (N, M, F)
    nbr_filter = total_gated_fea[:, :, 1:F]
    nbr_core = total_gated_fea[:, :, (F + 1):(2F)]
    nbr_filter = sigmoid.(nbr_filter)
    nbr_core = softplus.(nbr_core)
    # Sum over neighbors: (N, F)
    nbr_sumed = sum(nbr_filter .* nbr_core; dims=2)
    nbr_sumed = dropdims(nbr_sumed; dims=2)
    # BN2 over (N, F)
    Xbn2 = permutedims(nbr_sumed, (2, 1))  # (F, N)
    Ybn2, st_bn2 = layer.bn2(Xbn2, ps.bn2, st.bn2)
    nbr_sumed = permutedims(Ybn2, (2, 1))  # (N, F)
    # Residual + Softplus
    out = softplus.(atom_in_fea .+ nbr_sumed)
    new_st = (bn1=st_bn1, bn2=st_bn2)
    return out, new_st
end

struct Model{EMB,CONVS,C2FC,FCS,FCOUT,DO} <: AbstractLuxLayer
    classification::Bool
    n_targets::UInt32
    orig_atom_fea_len::UInt32
    nbr_fea_len::UInt32
    atom_fea_len::UInt32
    n_conv::UInt32
    h_fea_len::UInt32
    n_h::UInt32
    embedding::EMB
    convs::CONVS
    conv_to_fc::C2FC
    fcs::FCS
    fc_out::FCOUT
    dropout::DO
end
function Model(
    orig_atom_fea_len,
    nbr_fea_len;
    atom_fea_len=64,
    n_conv=3,
    h_fea_len=128,
    n_h=1,
    classification=false,
    n_targets=1,
    dropout_p=0.0,
)
    F = atom_fea_len
    d0 = orig_atom_fea_len
    B = nbr_fea_len
    H = h_fea_len
    embedding = Dense(d0 => F)
    convs = ntuple(_ -> ConvLayer(F, B), n_conv)
    conv_to_fc = Dense(F => H)
    fcs = n_h > 1 ? ntuple(_ -> Dense(H => H), n_h - 1) : ()
    out_dim = classification ? 2 : n_targets
    fc_out = Dense(H => out_dim)
    dropout = Dropout(dropout_p)
    return Model{
        typeof(embedding),
        typeof(convs),
        typeof(conv_to_fc),
        typeof(fcs),
        typeof(fc_out),
        typeof(dropout),
    }(
        classification,
        n_targets,
        d0,
        B,
        F,
        n_conv,
        H,
        n_h,
        embedding,
        convs,
        conv_to_fc,
        fcs,
        fc_out,
        dropout,
    )
end

function initialparameters(rng::AbstractRNG, m::Model)
    convs_ps = ntuple(i -> initialparameters(rng, m.convs[i]), m.n_conv)
    fcs_ps = m.n_h > 1 ? ntuple(i -> initialparameters(rng, m.fcs[i]), m.n_h - 1) : ()
    return (
        embedding=initialparameters(rng, m.embedding),
        convs=convs_ps,
        conv_to_fc=initialparameters(rng, m.conv_to_fc),
        fcs=fcs_ps,
        fc_out=initialparameters(rng, m.fc_out),
        dropout=initialparameters(rng, m.dropout),
    )
end

function initialstates(rng::AbstractRNG, m::Model)
    convs_st = ntuple(i -> initialstates(rng, m.convs[i]), m.n_conv)
    # Dense layers are stateless; Dropout may store RNG-related state depending on Lux version
    return (convs=convs_st, dropout=initialstates(rng, m.dropout))
end

"""
Forward pass for CrystalGraphConvNet.
Inputs:
- atom_fea:         (N, d0)
- nbr_fea:          (N, M, B)
- nbr_fea_idx:      (N, M)  -- 1-based
- crystal_atom_idx: Vector of index vectors
Returns:
- out: (N0, n_targets) for regression
       (N0, 2) log-probs for classification
"""
function (m::Model)((atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), ps, st)
    # 1) Embedding: (N, d0) -> (N, F)
    X = permutedims(atom_fea, (2, 1))  # (d0, N)
    Y, _ = m.embedding(X, ps.embedding, NamedTuple())
    atom_fea_h = permutedims(Y, (2, 1))  # (N, F)
    # 2) Convolutions
    conv_states = Vector{Any}(undef, m.n_conv)
    for i in 1:(m.n_conv)
        atom_fea_h, st_i = m.convs[i](
            (atom_fea_h, nbr_fea, nbr_fea_idx), ps.convs[i], st.convs[i]
        )
        conv_states[i] = st_i
    end
    new_convs_st = Tuple(conv_states)
    # 3) Pooling: (N, F) -> (N0, F)
    crys_fea = cgcnn_pooling(atom_fea_h, crystal_atom_idx)
    # 4) conv_to_fc + Softplus
    crys_fea = softplus.(crys_fea)
    Xc = permutedims(crys_fea, (2, 1))  # (F, N0)
    Yc, _ = m.conv_to_fc(Xc, ps.conv_to_fc, NamedTuple())
    crys_fea = permutedims(Yc, (2, 1))  # (N0, H)
    crys_fea = softplus.(crys_fea)
    # 5) Optional hidden layers after pooling
    if m.n_h > 1
        for i in 1:(m.n_h - 1)
            Xh = permutedims(crys_fea, (2, 1))  # (H, N0)
            Yh, _ = m.fcs[i](Xh, ps.fcs[i], NamedTuple())
            crys_fea = permutedims(Yh, (2, 1))
            crys_fea = softplus.(crys_fea)
        end
    end
    # 6) Dropout for classification
    new_dropout_st = st.dropout
    if m.classification
        crys_fea, new_dropout_st = m.dropout(crys_fea, ps.dropout, st.dropout)
    end
    # 7) Output
    Xo = permutedims(crys_fea, (2, 1))  # (H, N0)
    Yo, _ = m.fc_out(Xo, ps.fc_out, NamedTuple())
    out = permutedims(Yo, (2, 1))       # (N0, out_dim)
    if m.classification
        out = logsoftmax(out; dims=2)
    end
    new_st = (convs=new_convs_st, dropout=new_dropout_st)
    return out, new_st
end

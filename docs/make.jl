using CrystalGraphConvNet
using Documenter

DocMeta.setdocmeta!(CrystalGraphConvNet, :DocTestSetup, :(using CrystalGraphConvNet); recursive=true)

makedocs(;
    modules=[CrystalGraphConvNet],
    authors="singularitti <singularitti@outlook.com> and contributors",
    sitename="CrystalGraphConvNet.jl",
    format=Documenter.HTML(;
        canonical="https://MaterialsRealm.github.io/CrystalGraphConvNet.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MaterialsRealm/CrystalGraphConvNet.jl",
    devbranch="main",
)

push!(LOAD_PATH,"../src/")

using MicroMacro
using Documenter

makedocs(modules=[MicroMacro],
         doctest = false,
         format = :html,
         sitename = "MicroMacro.jl",
         pages = ["Documentation"    => "index.md"])

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com:JuliaVlasov/MicroMacro.jl.git",
 )

using Documenter
using MicroMacro

makedocs(
    sitename = "MicroMacro",
    format = :html,
    modules = [MicroMacro]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com:JuliaVlasov/MicroMacro.jl.git",
)

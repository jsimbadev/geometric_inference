# Approximation of the Tangent Bundle

The goal of this research is to develop data structures and algorithms that allow piecewise construction of implied data manifolds by linearizations.
The motivation is directly from the differential geometric definition of a manifold: a collection of open sets with bijections into Euclidean space.

## GeomDiagnostics CLI

`GeomDiagnostics` exposes a Comonicon CLI entrypoint from Julia code.

### Run directly from Julia (no installed binary)

From the package directory:

```bash
cd GeomDiagnostics
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -e "using GeomDiagnostics; GeomDiagnostics.command_main()" -- normal 100 --seed 999
```

Equivalent command from repo root:

```bash
julia --project=GeomDiagnostics -e "using GeomDiagnostics; GeomDiagnostics.command_main()" -- normal 100 --seed 999
```

### Build/install a clean CLI executable

Use the wrapper script in this repo:

```bash
./scripts/build_cli.sh
```

This installs a CLI executable at:

```bash
./.cli/bin/geomdiagnostics
```

Then run it directly:

```bash
./.cli/bin/geomdiagnostics normal 100 --seed 999
```

You can override install location and binary name:

```bash
./scripts/build_cli.sh /tmp/my-cli gd
# -> /tmp/my-cli/bin/gd
```

### Comonicon build notes

For the Comonicon version currently in this project (`v1.0.8`), the build/install entrypoint is:

```bash
julia --project=GeomDiagnostics -e "using GeomDiagnostics; GeomDiagnostics.comonicon_install()"
```

If you have seen examples like `Comonicon.build()`, that API is not available in this installed version.
`comonicon_install()` is the supported way to generate the installed command wrapper.

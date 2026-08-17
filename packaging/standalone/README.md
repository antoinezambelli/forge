# Standalone Forge Proxy builds

All three targets use `forge_proxy.spec` through the Python 3.12 build driver.
The driver rejects non-native target requests, builds and fully smokes `onedir`
before allowing `onefile`, and writes artifact-derived `evidence.json` beside
each generated payload under the ignored `standalone-dist/` directory. A fully
passing two-form build writes `selection.json` choosing the onefile payload.

## Windows x64

From a PowerShell prompt at the repository root:

```powershell
.\scripts\standalone\build_windows.ps1
```

This recreates `.standalone-build-env` with only the project `anthropic` extra
and PyInstaller, then builds and smokes both forms. The packaged smoke runs the
executable from an unrelated temporary directory with Python path variables
removed and checks version, help, health, OpenAI-shaped forwarding,
Anthropic-shaped SDK forwarding, `CTRL_BREAK_EVENT` shutdown, listener closure,
and onefile extraction cleanup.

## Linux x64 / glibc 2.35

Build natively in the Ubuntu 22.04 image (Docker output can be copied from the
container's `/forge/standalone-dist` directory):

```sh
docker build -f packaging/standalone/linux/Dockerfile -t forge-proxy-linux .
docker run --name forge-proxy-linux-build forge-proxy-linux
```

The completed artifact inspection checks every ELF object in the onedir bundle,
the onefile launcher, and every ELF object recorded in its collection inventory,
failing if any referenced GLIBC symbol exceeds 2.35.

## macOS arm64

On an arm64 Mac with Python 3.12:

```sh
./scripts/standalone/build_macos.sh
```

Linux x64 and macOS arm64 definitions are authored for native execution. They
are not verified by a Windows build run.

Generated payloads and evidence are local build output. This workflow does not
publish release assets, tags, or remote state.

## Release automation and evidence

`proxy-release-candidate.yml` builds the Windows x64, Ubuntu 22.04 Linux x64,
and macOS arm64 targets once on their native runners. Each selected onefile is
run through packaged smoke and the isolated install/init/check/same-version
repair/uninstall lifecycle. The Ubuntu-built Linux archive is then executed
unchanged on Ubuntu 22.04, Debian 12, and Fedora 44. Permission-preserving
archives carry the selected byte, its SHA-256 identity, size, version, and the
portable cold-start, extraction/layout, dependency/GLIBC, packaged-smoke, and
lifecycle evidence. The optional `real_backends` input adds the separate
llama-server, Ollama, and vLLM sanity pass on a self-hosted
`proxy-real-backends` runner with `FORGE_PROXY_GGUF` and
`FORGE_PROXY_VLLM_URL`; models and GPUs are not part of the default gate.

`proxy-release.yml` must be manually dispatched from an existing exact
`refs/tags/vX.Y.Z` whose version matches `pyproject.toml` and whose GitHub
Release already exists. It rebuilds no selected byte after testing. The three
native outputs and all Linux compatibility results gate one immutable staging
job. One environment-gated publication job re-hashes that staging archive,
adds free GitHub build-provenance attestations for the three executables, and
uploads to the existing exact Release. The checksum file precedes the manifest;
`proxy-X.Y.Z.json` is uploaded last as the completeness marker. Publication
rejects existing Proxy names and rolls back only assets journaled by that run.
The Release's `target_commitish` is recorded for information, not used as tag
identity.

`proxy-pointer.yml` only validates `installer/proxy-stable.txt`. Its intentional
absence before the first public Proxy release passes without a network lookup.
If present, it must contain one bare `X.Y.Z` line and reference an existing,
complete exact manifest. None of these workflows creates or advances it.

## Mould-owned human release handoff

The combined procedure remains outside Forge and is performed in this order:

1. Require the Proxy release-candidate gate (and the opt-in real-backend pass
   when the operator calls for it).
2. Follow the existing Forge PyPI release recipe.
3. Dispatch the exact-tag Proxy workflow from that same Forge tag.
4. Require complete manifest-last publication and all three published exact
   install checks to pass.
5. Human-review, create or advance, and commit the stable pointer in a later
   change.

An ordinary Forge/PyPI/GitHub release may omit Proxy artifacts. This
implementation run does not dispatch workflows, create tags, publish, upload,
attest, edit a Release, or create/advance the stable pointer.

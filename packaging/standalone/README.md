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

Changing `installer/proxy-stable.txt` in a pull request declares that the Forge
release is also a Proxy release and triggers `proxy-release-candidate.yml`.
The workflow exposes exactly three jobs: Windows x64, Linux x64, and macOS.
Each job runs its public bootstrap contracts, builds the native artifact, and
exercises packaged smoke plus the isolated
install/init/check/same-version-repair/uninstall lifecycle. The Linux job also
executes the same Ubuntu-built bytes sequentially on Ubuntu 22.04, Debian 12,
and Fedora 44. An ordinary Forge release leaves the pointer unchanged and does
not run Proxy CI.

The Proxy pointer and `pyproject.toml` must contain the same version in a Proxy
release pull request. Permission-preserving archives carry the selected byte,
its SHA-256 identity, size, version, and the portable cold-start,
extraction/layout, dependency/GLIBC, packaged-smoke, and lifecycle evidence.

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

## Mould-owned human release handoff

The combined procedure remains outside Forge and is performed in this order:

1. In the release pull request, bump `pyproject.toml` and
   `installer/proxy-stable.txt` to the same version.
2. Require all three Proxy release-candidate jobs to pass.
3. Follow the existing Forge PyPI release recipe.
4. Dispatch the exact-tag Proxy workflow from that same Forge tag.
5. Require complete manifest-last publication and all three published exact
   install checks to pass. No later pointer change is required.

An ordinary Forge/PyPI/GitHub release may omit Proxy artifacts by leaving the
pointer unchanged. This implementation run does not dispatch workflows, create
tags, publish, upload, attest, or edit a Release.

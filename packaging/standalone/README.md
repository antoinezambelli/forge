# Standalone Forge Proxy builds

All three targets use `forge_proxy.spec` through the Python 3.14 build driver.
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

## Linux x64 / glibc 2.39

Build natively in the Ubuntu 24.04 image (Docker output can be copied from the
container's `/forge/standalone-dist` directory):

```sh
docker build -f packaging/standalone/linux/Dockerfile -t forge-proxy-linux .
docker run --name forge-proxy-linux-build forge-proxy-linux
```

The completed artifact inspection checks every ELF object in the onedir bundle,
the onefile launcher, and every ELF object recorded in its collection inventory,
failing if any referenced GLIBC symbol exceeds 2.39.

## macOS arm64

On an arm64 Mac with Python 3.14:

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
Three independent platform jobs build Windows x64, Linux x64, and macOS ARM64
through their documented release entrypoints. Each exercises packaged smoke
plus the isolated install/init/check/reinstall/uninstall lifecycle. When an
older standalone Proxy artifact is retrievable, each job additionally tests a
forward update and exact-version reinstall; otherwise only those two
cross-version checks are skipped. The Linux job also extracts the Docker-built
artifact and executes those same bytes sequentially on Ubuntu 24.04, Debian 13,
and Fedora 43. A fourth aggregation-only job combines the three passing outputs
into one retained release-candidate artifact. An ordinary Forge release leaves
the pointer unchanged and does not run Proxy CI.

The Proxy pointer and `pyproject.toml` must contain the same version in a Proxy
release pull request. Permission-preserving archives carry the selected byte,
its SHA-256 identity, size, version, and the portable cold-start,
extraction/layout, dependency/GLIBC, packaged-smoke, and lifecycle evidence.

`proxy-release.yml` is manually dispatched from protected `main` with an
existing exact `vX.Y.Z` Forge tag, its GitHub Release, and the successful
candidate workflow run ID. It verifies that the tag has the same source tree as
the retained candidate, re-hashes that candidate, adds free GitHub
build-provenance attestations for the three executables, and uploads those exact
bytes without rebuilding or repeating lifecycle tests. The checksum file
precedes the manifest; `proxy-X.Y.Z.json` is uploaded last as the completeness
marker. Publication rejects existing Proxy names and rolls back only assets
journaled by that run. The Release's `target_commitish` is recorded for
information, not used as tag identity.

## Mould-owned human release handoff

The combined procedure remains outside Forge and is performed in this order:

1. In the release pull request, bump `pyproject.toml` and
   `installer/proxy-stable.txt` to the same version.
2. Require all three Proxy release-candidate jobs to pass and retain that run
   ID.
3. Follow the existing Forge PyPI release recipe.
4. Dispatch the Proxy publication workflow from protected `main` with the exact
   Forge tag and candidate run ID.
5. Require complete manifest-last publication. No later pointer change is
   required.

An ordinary Forge/PyPI/GitHub release may omit Proxy artifacts by leaving the
pointer unchanged. This implementation run does not dispatch workflows, create
tags, publish, upload, attest, or edit a Release.

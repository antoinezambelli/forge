# Install Forge Proxy

Forge Proxy is available as a standalone, self-contained command. The bundle
contains Forge, a private Python runtime, Forge's core dependencies, and the
Anthropic SDK. It does **not** install a backend executable, model, GPU or driver
stack, service, credentials, or client configuration.

Install and operate a downstream backend separately. See [Backend
Setup](BACKEND_SETUP.md) for backend installation and the [Proxy Server
overview](../README.md#proxy-server) and [User Guide](USER_GUIDE.md) for Proxy
behavior and backend selection.

## Release availability

The versionless installers resolve
`installer/proxy-stable.txt`. If that pointer is absent, they report that no
stable standalone Proxy release has been published. This is conditional:
ordinary Forge, PyPI, and GitHub releases may omit the standalone Proxy assets.
Other pointer fetch failures are reported as unavailable downloads rather than
as an unpublished release.

An exact `X.Y.Z` install works only when the exact `vX.Y.Z` Forge Release
contains the complete Proxy artifact set. The examples below do not imply that
any particular stable or exact standalone release has been published.

## Supported hosts and prerequisites

| Host | Native target | Release artifact | Prerequisites |
|---|---|---|---|
| Windows x64 | `windows-x86_64` | `forge-proxy-windows-x86_64.exe` | PowerShell |
| Linux x64 | `linux-x86_64-gnu` | `forge-proxy-linux-x86_64-gnu` | GNU libc 2.35 or newer, with `ldd`; `curl`; `mktemp`; `sha256sum` |
| macOS arm64 | `macos-arm64` | `forge-proxy-macos-arm64` | `curl`; `mktemp`; `shasum -a 256` |

Other operating systems, architectures, and libc combinations are unsupported
and fail closed. There is no fallback to pip.

## Install with the public bootstrap

### One command, versionless

These commands use the stable pointer and succeed only while that pointer
exists.

POSIX:

```sh
curl -fsSL https://raw.githubusercontent.com/antoinezambelli/forge/main/install.sh | sh
```

Installation never consumes onboarding input. After either bootstrap finishes,
open a refreshed terminal and run `forge-proxy init`, then `forge-proxy check`.

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/antoinezambelli/forge/main/install.ps1 | iex
```

### Save, inspect, then execute

POSIX:

```sh
curl -fsSLo install.sh https://raw.githubusercontent.com/antoinezambelli/forge/main/install.sh
less install.sh
sh install.sh
```

Windows PowerShell:

```powershell
iwr https://raw.githubusercontent.com/antoinezambelli/forge/main/install.ps1 -OutFile install.ps1
Get-Content .\install.ps1
.\install.ps1
```

The saved scripts accept exact version, no-init, and absolute custom-root
options. This example combines all three:

```text
install.sh [--version X.Y.Z] [--no-init] [--install-root ABSOLUTE]
install.ps1 [-Version X.Y.Z] [-NoInit] [-InstallRoot ABSOLUTE]
```

```sh
sh install.sh --version X.Y.Z --no-init --install-root "$HOME/.local/share/forge-proxy-custom"
```

```powershell
.\install.ps1 -Version X.Y.Z -NoInit -InstallRoot "$env:LOCALAPPDATA\Forge Proxy Custom"
```

`--no-init`/`-NoInit` remains accepted for explicit automation, although all
installations now leave initialization to the subsequent `forge-proxy init`
command. Both scripts select the native target, download `proxy-X.Y.Z.json` and
its declared artifact, verify the declared byte size and SHA-256 digest, then
hand the verified digest to the artifact's `install-artifact` command.
Unsupported hosts stop before download.

## Manual immutable artifact handoff

Use this path when another system downloads or transfers release assets. First
download `proxy-X.Y.Z.json` from the exact `vX.Y.Z` Forge Release. In that
manifest, find the entry for the native target from the table above and copy its
`name`, `size`, and `sha256` values. Download the manifest-provided filename and
measure and hash those exact bytes. Pass the same verified digest to
`install-artifact`.

The following POSIX example also shows an exact custom-root install or recovery.
Substitute the four uppercase values from the native manifest entry. On macOS,
replace the `sha256sum` line with
`test "$(shasum -a 256 "$artifact" | awk '{print $1}')" = "$sha256"`.

```sh
version=X.Y.Z
artifact=ARTIFACT_NAME_FROM_THE_TARGET_ENTRY
size=SIZE_FROM_THE_TARGET_ENTRY
sha256=SHA256_FROM_THE_TARGET_ENTRY
root="$HOME/.local/share/forge-proxy-custom"
curl -fsSLO "https://github.com/antoinezambelli/forge/releases/download/v$version/proxy-$version.json"
curl -fsSLO "https://github.com/antoinezambelli/forge/releases/download/v$version/$artifact"
test "$(wc -c < "$artifact" | tr -d ' ')" = "$size"
printf '%s  %s\n' "$sha256" "$artifact" | sha256sum -c -
chmod +x "./$artifact"
"./$artifact" install-artifact --version "$version" --sha256 "$sha256" --no-init --install-root "$root"
```

The equivalent Windows PowerShell handoff is:

```powershell
$Version = 'X.Y.Z'
$Artifact = 'ARTIFACT_NAME_FROM_THE_TARGET_ENTRY'
$Size = SIZE_FROM_THE_TARGET_ENTRY
$Sha256 = 'SHA256_FROM_THE_TARGET_ENTRY'
$Root = "$env:LOCALAPPDATA\Forge Proxy Custom"
iwr "https://github.com/antoinezambelli/forge/releases/download/v$Version/proxy-$Version.json" -OutFile "proxy-$Version.json"
iwr "https://github.com/antoinezambelli/forge/releases/download/v$Version/$Artifact" -OutFile $Artifact
if ((Get-Item -LiteralPath $Artifact).Length -ne $Size) { throw 'artifact size mismatch' }
if ((Get-FileHash -LiteralPath $Artifact -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256.ToLowerInvariant()) { throw 'artifact checksum mismatch' }
& ".\$Artifact" install-artifact --version $Version --sha256 $Sha256 --no-init --install-root $Root
```

`--no-init` and the absolute custom root are optional for a new installation.
For recovery of an installation originally created at a custom root, they are
not interchangeable: every exact bootstrap or manual handoff must repeat the
same `--install-root`/`-InstallRoot` value. Omitting the original root targets
the platform default and does not recover the custom-root installation; rerun
with the original absolute root. Do not treat the resulting locations as
multiple supported active installations.

For example, exact bootstrap recovery at the roots used above is:

```sh
sh install.sh --version X.Y.Z --no-init --install-root "$HOME/.local/share/forge-proxy-custom"
```

```powershell
.\install.ps1 -Version X.Y.Z -NoInit -InstallRoot "$env:LOCALAPPDATA\Forge Proxy Custom"
```

## Create a profile or launch with flags

Installation prints these as the next steps. Create the default profile, or
create another named profile, with:

```console
forge-proxy init
forge-proxy init --profile local-openai
```

`init` preserves supplied string values exactly and prints the corresponding
`forge-proxy --profile NAME` launch command after writing the profile.

For a noninteractive unmanaged OpenAI-shaped backend:

```console
forge-proxy init --profile local-openai --non-interactive --backend-url http://127.0.0.1:8000 --backend openai
```

For a noninteractive unmanaged Anthropic-shaped backend, select it explicitly:

```console
forge-proxy init --profile anthropic-gateway --non-interactive --backend-url https://gateway.example --backend anthropic
```

Managed profile TOML uses `schema_version = 1`. CLI hyphenated names map to
underscore TOML keys, such as `--backend-url` to `backend_url`. Profile fields
and CLI flags share meanings, defaults, applicability, and canonical
validation. Managed profile files are sparse: omitted defaults are applied at
load time instead of being written.

A profile or config is one complete configuration source, not an overlay.
`--profile` and `--config` are mutually exclusive, and either selector is also
mutually exclusive with all Proxy configuration flags. Multiple configuration
flags may—and often must—be combined in flag-only mode.

These are three separate, valid launch shapes:

```console
forge-proxy --profile local-openai
```

```console
forge-proxy --backend-url http://127.0.0.1:8000 --backend openai --host 127.0.0.1 --port 8081
```

```console
forge-proxy --backend-url https://gateway.example --backend anthropic --model claude-route --port 8081
```

Use `forge-proxy --config /absolute/path/to/profile.toml` when another tool owns
the complete TOML file; Forge reads but does not rewrite it.

## Check, update, recover, and uninstall

After at least one managed profile exists, validate the private runtime, all
managed profiles, and a local Forge health listener:

```console
forge-proxy check
```

Updates are forward-only. With no option, `update` follows the stable pointer;
an exact option selects a newer exact release:

```console
forge-proxy update
forge-proxy update --version X.Y.Z
```

`update` does not install a lower version. For an exact reinstall, lower-version
recovery, or recovery when the installed command cannot run, use the external
bootstrap or manual artifact handoff above. For a custom-root installation,
repeat its original absolute `--install-root`/`-InstallRoot` on every recovery
command; omitting it targets the platform default and does not recover the
custom-root installation.

To remove the managed installation:

```console
forge-proxy uninstall
```

This delegates to the installation's owned native uninstaller. It removes only
owned installation state and PATH/shell integration; it does not uninstall a
backend, model, driver, or other independently managed software.

## Filesystem and PATH behavior

| Host | Default install root | Command directory | Managed profile root |
|---|---|---|---|
| Windows | `%LOCALAPPDATA%\Forge` | `%LOCALAPPDATA%\Forge\bin` | `%APPDATA%\Forge\profiles` |
| Linux | `${XDG_DATA_HOME:-$HOME/.local/share}/forge` | `$HOME/.local/bin` | `${XDG_CONFIG_HOME:-$HOME/.config}/forge/profiles` |
| macOS | `$HOME/Library/Application Support/Forge` | `$HOME/.local/bin` | `$HOME/Library/Application Support/Forge/profiles` |
| Custom absolute root | `<root>` | `<root>/bin` | The host default above |

Artifacts occupy immutable `<root>/versions/X.Y.Z/` slots. Installation state,
an ownership marker, the current command, and the native uninstaller record the
owned installation. After an update, the current version and at most one prior
version slot are retained.

On Windows, Forge owns one exact user-PATH entry for the command directory. On
bash and zsh, it owns one marked startup-file block and reports the startup file
it changed plus how to undo the change. For an unknown POSIX shell, the
installer prints an `export PATH=...` instruction instead. Open a refreshed
terminal or reload the shell startup file before expecting `forge-proxy` to be
found.

## Release identity and integrity

Forge Proxy uses the exact Forge `X.Y.Z` version, `vX.Y.Z` tag, and Forge
Release namespace. It has no separate Proxy semantic version and no moving
Proxy tag. A complete standalone release contains one ruled artifact for each
supported target plus `proxy-X.Y.Z.json` and `proxy-X.Y.Z.sha256`.

The exact manifest declares every artifact filename, byte size, and SHA-256
digest. The bootstraps verify size and digest before invoking the artifact, and
`install-artifact` verifies the same digest again. Releases may also carry free
GitHub build-provenance attestations for optional independent verification; no
attestation verifier is required at install time.

The stable pointer contains one exact `X.Y.Z` value and resolves to that exact
release manifest. Advancing the pointer selects another immutable Forge
Release; it does not change a tag or artifact in place. If the pointer is
absent, only complete exact releases can be targeted, and only by version.

## Generic noninteractive wrapper

This generic shell flow installs one exact release without prompts, creates an
unmanaged named profile, checks the installation, and starts with that profile:

```sh
version=X.Y.Z
root="$HOME/.local/share/forge-proxy-custom"
sh ./install.sh --version "$version" --no-init --install-root "$root"
"$root/bin/forge-proxy" init --profile local-openai --non-interactive --backend-url http://127.0.0.1:8000 --backend openai
"$root/bin/forge-proxy" check
"$root/bin/forge-proxy" --profile local-openai
```

## Further reading

- [Backend Setup](BACKEND_SETUP.md) — install and run downstream backends.
- [README: Proxy Server](../README.md#proxy-server) — behavior and common launch context.
- [User Guide](USER_GUIDE.md) — detailed Proxy and backend behavior.
- `forge-proxy --help` — installed CLI source-selection, lifecycle, and documentation links.

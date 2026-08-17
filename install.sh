#!/bin/sh
# Forge Proxy bootstrap installer for Linux x64/glibc 2.35+ and macOS arm64.
# One line: curl -fsSL https://raw.githubusercontent.com/antoinezambelli/forge/main/install.sh | sh
# Save, inspect, execute:
#   curl -fsSLo install.sh https://raw.githubusercontent.com/antoinezambelli/forge/main/install.sh
#   less install.sh
#   sh install.sh --version X.Y.Z --no-init --install-root '/opt/forge proxy'
# Manual immutable install:
#   curl -fsSLO https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/proxy-X.Y.Z.json
#   artifact=ARTIFACT_NAME_FROM_THE_TARGET_ENTRY
#   curl -fsSLO "https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/$artifact"
#   # Compare its manifest byte size and verify with sha256sum or shasum -a 256.
#   chmod +x "./$artifact"
#   "./$artifact" install-artifact --version X.Y.Z --sha256 HEX

set -eu

show_help() {
    cat <<'EOF'
Usage: install.sh [--version X.Y.Z] [--no-init] [--install-root ABSOLUTE] [--help]

Downloads, verifies, and hands a supported release to its install-artifact
operation. With no version, the stable pointer is used. Installation never
prompts; --no-init remains accepted for explicit automation.

One line:
  curl -fsSL https://raw.githubusercontent.com/antoinezambelli/forge/main/install.sh | sh

Save, inspect, execute:
  curl -fsSLo install.sh https://raw.githubusercontent.com/antoinezambelli/forge/main/install.sh
  less install.sh
  sh install.sh --version X.Y.Z --no-init --install-root '/opt/forge proxy'

Manual immutable install:
  curl -fsSLO https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/proxy-X.Y.Z.json
  artifact=ARTIFACT_NAME_FROM_THE_TARGET_ENTRY
  curl -fsSLO "https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/$artifact"
  # Compare its manifest byte size and verify with sha256sum or shasum -a 256.
  chmod +x "./$artifact"
  "./$artifact" install-artifact --version X.Y.Z --sha256 HEX
EOF
}

die() {
    printf 'forge-proxy bootstrap: %s\n' "$1" >&2
    exit 1
}

valid_version() {
    printf '%s\n' "$1" | grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
}

version=
no_init=0
install_root=
seen_version=0
seen_no_init=0
seen_root=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            [ "$seen_version" -eq 0 ] || die 'duplicate argument: --version'
            seen_version=1
            shift
            [ "$#" -gt 0 ] || die '--version requires X.Y.Z'
            version=$1
            ;;
        --no-init)
            [ "$seen_no_init" -eq 0 ] || die 'duplicate argument: --no-init'
            seen_no_init=1
            no_init=1
            ;;
        --install-root)
            [ "$seen_root" -eq 0 ] || die 'duplicate argument: --install-root'
            seen_root=1
            shift
            [ "$#" -gt 0 ] || die '--install-root requires an absolute path'
            install_root=$1
            ;;
        --help)
            [ "$#" -eq 1 ] || die '--help cannot be combined with other arguments'
            show_help
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done

[ -z "$version" ] || valid_version "$version" || die "invalid Proxy version: '$version'; expected X.Y.Z"
case "$install_root" in
    ''|/*) ;;
    *) die '--install-root must be an absolute path' ;;
esac

testing=${_FORGE_PROXY_BOOTSTRAP_TESTING:-0}
if [ "$testing" = 1 ]; then
    system=${_FORGE_PROXY_BOOTSTRAP_SYSTEM:-}
    machine=${_FORGE_PROXY_BOOTSTRAP_MACHINE:-}
else
    system=$(uname -s)
    machine=$(uname -m)
fi

case "$system:$machine" in
    Darwin:arm64|Darwin:aarch64) target=macos-arm64 ;;
    Linux:x86_64|Linux:amd64)
        if [ "$testing" = 1 ]; then
            libc_banner=${_FORGE_PROXY_BOOTSTRAP_LDD_OUTPUT:-}
        else
            command -v ldd >/dev/null 2>&1 || die 'unsupported Linux libc: ldd is unavailable'
            libc_banner=$(ldd --version 2>&1) || die 'unsupported Linux libc: ldd --version failed'
        fi
        printf '%s\n' "$libc_banner" | grep -Eiq '(GNU libc|GLIBC|GNU C Library)' ||
            die 'unsupported Linux libc: GNU libc/glibc could not be proven'
        libc_version=$(printf '%s\n' "$libc_banner" | sed -nE 's/.* ([0-9]+)\.([0-9]+)([^0-9].*)?$/\1 \2/p' | sed -n '1p')
        [ -n "$libc_version" ] || die 'unsupported Linux libc: glibc version is unknown'
        libc_major=${libc_version% *}
        libc_minor=${libc_version#* }
        if [ "$libc_major" -lt 2 ] || { [ "$libc_major" -eq 2 ] && [ "$libc_minor" -lt 35 ]; }; then
            die 'unsupported Linux libc: glibc 2.35 or newer is required'
        fi
        target=linux-x86_64-gnu
        ;;
    *) die "unsupported standalone target: $system $machine" ;;
esac

pointer_url=https://raw.githubusercontent.com/antoinezambelli/forge/main/installer/proxy-stable.txt
release_base=https://github.com/antoinezambelli/forge/releases/download
temp_base=${TMPDIR:-/tmp}
if [ "$testing" = 1 ]; then
    pointer_url=${_FORGE_PROXY_BOOTSTRAP_POINTER_URL:-$pointer_url}
    release_base=${_FORGE_PROXY_BOOTSTRAP_RELEASE_BASE_URL:-$release_base}
    temp_base=${_FORGE_PROXY_BOOTSTRAP_TEMP_ROOT:-$temp_base}
fi
release_base=${release_base%/}

command -v curl >/dev/null 2>&1 || die 'curl is required'
temporary=$(mktemp -d "$temp_base/forge-proxy-bootstrap.XXXXXX") || die 'cannot create temporary directory'
cleanup() { rm -rf -- "$temporary"; }
trap cleanup EXIT HUP INT TERM

fetch() {
    curl --fail --location --silent --show-error --output "$2" "$1"
}

fetch_pointer() {
    pointer_status=0
    pointer_http=$(curl --fail --location --silent --show-error \
        --write-out '%{http_code}' --output "$2" "$1") || pointer_status=$?
    [ "$pointer_status" -eq 0 ] && return 0
    [ "$pointer_http" = 404 ] && return 44
    return 1
}

if [ -z "$version" ]; then
    if fetch_pointer "$pointer_url" "$temporary/proxy-stable.txt"; then
        :
    else
        pointer_status=$?
        [ "$pointer_status" -eq 44 ] &&
            die 'no stable standalone Proxy release has been published'
        die "download unavailable: $pointer_url"
    fi
    version=$(cat "$temporary/proxy-stable.txt")
    pointer_size=$(wc -c < "$temporary/proxy-stable.txt" | tr -d ' ')
    version_size=${#version}
    valid_version "$version" &&
        { [ "$pointer_size" -eq "$version_size" ] || [ "$pointer_size" -eq $((version_size + 1)) ]; } ||
            die 'stable pointer must contain one bare X.Y.Z line'
fi

manifest="$temporary/proxy-$version.json"
manifest_url="$release_base/v$version/proxy-$version.json"
fetch "$manifest_url" "$manifest" || die "download unavailable: $manifest_url"
compact=$(tr -d ' \t\r\n' < "$manifest")
manifest_version=$(printf '%s\n' "$compact" | sed -n 's/^{"version":"\([^"]*\)","artifacts":{.*}}$/\1/p')
artifacts=$(printf '%s\n' "$compact" | sed -n 's/^{"version":"[^"]*","artifacts":{\(.*\)}}$/\1/p')
if [ -z "$manifest_version" ]; then
    manifest_version=$(printf '%s\n' "$compact" | sed -n 's/^{"artifacts":{.*},"version":"\([^"]*\)"}$/\1/p')
    artifacts=$(printf '%s\n' "$compact" | sed -n 's/^{"artifacts":{\(.*\)},"version":"[^"]*"}$/\1/p')
fi
[ -n "$manifest_version" ] || die 'release manifest must contain only version and artifacts'
[ "$manifest_version" = "$version" ] || die 'release manifest version does not match the requested version'
[ -n "$artifacts" ] || die 'release manifest contains no artifacts'

selection="$temporary/selection"
printf '%s\n' "$artifacts" | sed 's/},"/}\
"/g' > "$temporary/entries"
while IFS= read -r manifest_entry; do
    entry_target=$(printf '%s\n' "$manifest_entry" | sed -n 's/^"\([^"]*\)":{.*}$/\1/p')
    entry_body=$(printf '%s\n' "$manifest_entry" | sed -n 's/^"[^"]*":{\(.*\)}$/\1/p')
    case "$entry_target" in
        windows-x86_64|linux-x86_64-gnu|macos-arm64) ;;
        *) die "unsupported release target: $entry_target" ;;
    esac
    entry_name=
    entry_sha=
    entry_size=
    field_count=0
    old_ifs=$IFS
    IFS=,
    for field in $entry_body; do
        field_count=$((field_count + 1))
        case "$field" in
            '"name":"'*) entry_name=$(printf '%s\n' "$field" | sed -n 's/^"name":"\([^"]*\)"$/\1/p') ;;
            '"sha256":"'*) entry_sha=$(printf '%s\n' "$field" | sed -n 's/^"sha256":"\([^"]*\)"$/\1/p') ;;
            '"size":'*) entry_size=$(printf '%s\n' "$field" | sed -n 's/^"size":\([0-9][0-9]*\)$/\1/p') ;;
            *) die "invalid release manifest entry for $entry_target" ;;
        esac
    done
    IFS=$old_ifs
    [ "$field_count" -eq 3 ] && [ -n "$entry_name" ] && [ -n "$entry_sha" ] && [ -n "$entry_size" ] ||
        die "invalid release manifest entry for $entry_target"
    printf '%s\n' "$entry_name" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*$' ||
        die "invalid release manifest entry for $entry_target"
    [ "$entry_name" != . ] && [ "$entry_name" != .. ] || die "invalid release manifest entry for $entry_target"
    printf '%s\n' "$entry_sha" | grep -Eq '^[0-9a-fA-F]{64}$' ||
        die "invalid release manifest entry for $entry_target"
    if [ "$entry_target" = "$target" ]; then
        printf '%s\n%s\n%s\n' "$entry_name" "$entry_sha" "$entry_size" > "$selection"
    fi
done < "$temporary/entries"

[ -f "$selection" ] || die "release manifest has no artifact for $target"
artifact_name=$(sed -n '1p' "$selection")
expected_sha=$(sed -n '2p' "$selection" | tr 'A-F' 'a-f')
expected_size=$(sed -n '3p' "$selection")
artifact="$temporary/$artifact_name"
artifact_url="$release_base/v$version/$artifact_name"
fetch "$artifact_url" "$artifact" || die "download unavailable: $artifact_url"
actual_size=$(wc -c < "$artifact" | tr -d ' ')
[ "$actual_size" = "$expected_size" ] || die 'downloaded artifact size does not match release manifest'
case "$target" in
    macos-arm64)
        command -v shasum >/dev/null 2>&1 || die 'shasum is required'
        actual_sha=$(shasum -a 256 "$artifact" | awk '{print $1}')
        ;;
    *)
        command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
        actual_sha=$(sha256sum "$artifact" | awk '{print $1}')
        ;;
esac
[ "$actual_sha" = "$expected_sha" ] || die 'downloaded artifact checksum mismatch'
chmod +x "$artifact"

set -- install-artifact --version "$version" --sha256 "$expected_sha"
[ "$no_init" -eq 0 ] || set -- "$@" --no-init
[ -z "$install_root" ] || set -- "$@" --install-root "$install_root"
if "$artifact" "$@"; then
    handoff_status=0
else
    handoff_status=$?
fi
exit "$handoff_status"

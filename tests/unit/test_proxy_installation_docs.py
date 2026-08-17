"""Contract and local-fixture checks for the canonical Proxy installation page."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from forge.proxy import __main__ as proxy_cli
from forge.proxy import _profiles
from forge.proxy._config import _normalize_proxy_config
from forge.proxy._options import supplied_proxy_options
from scripts.standalone import inputs, release


ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "PROXY_INSTALLATION.md"
PAGE_TEXT = PAGE.read_text(encoding="utf-8")


def test_single_canonical_page_is_discoverable_from_readme_and_help() -> None:
    assert [path.relative_to(ROOT) for path in ROOT.rglob("PROXY_INSTALLATION.md")] == [
        Path("docs/PROXY_INSTALLATION.md")
    ]
    assert "[Forge Proxy Installation](docs/PROXY_INSTALLATION.md)" in (
        ROOT / "README.md"
    ).read_text(encoding="utf-8")
    help_text = proxy_cli._build_parser().format_help()
    for url in (
        "https://github.com/antoinezambelli/forge/blob/main/docs/PROXY_INSTALLATION.md",
        "https://github.com/antoinezambelli/forge#proxy-server",
        "https://github.com/antoinezambelli/forge/blob/main/docs/USER_GUIDE.md",
    ):
        assert url in help_text


def test_target_artifact_and_path_claims_come_from_product_contracts() -> None:
    for target in inputs.SUPPORTED_TARGETS:
        assert f"`{target}`" in PAGE_TEXT
        assert f"`{release.artifact_name(target)}`" in PAGE_TEXT

    for claim in (
        "%LOCALAPPDATA%\\Forge",
        "%APPDATA%\\Forge\\profiles",
        "${XDG_DATA_HOME:-$HOME/.local/share}/forge",
        "${XDG_CONFIG_HOME:-$HOME/.config}/forge/profiles",
        "$HOME/Library/Application Support/Forge",
        "<root>/bin",
    ):
        assert claim in PAGE_TEXT


def test_documented_option_spellings_match_script_and_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shell = (ROOT / "install.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "install.ps1").read_text(encoding="utf-8")
    for line in (
        "artifact=ARTIFACT_NAME_FROM_THE_TARGET_ENTRY",
        'curl -fsSLO "https://github.com/antoinezambelli/forge/releases/download/vX.Y.Z/$artifact"',
        'chmod +x "./$artifact"',
        '"./$artifact" install-artifact --version X.Y.Z --sha256 HEX',
    ):
        assert shell.count(line) == 2
    shell_usage = re.search(r"Usage: install\.sh \[(.*?)\] \[(.*?)\] \[(.*?)\]", shell)
    ps_usage = re.search(r"Usage: install\.ps1 \[(.*?)\] \[(.*?)\] \[(.*?)\]", powershell)
    assert shell_usage is not None and ps_usage is not None
    for spelling in ("--version X.Y.Z", "--no-init", "--install-root ABSOLUTE"):
        assert spelling in shell_usage.group(0)
        assert spelling in PAGE_TEXT
    for spelling in ("-Version X.Y.Z", "-NoInit", "-InstallRoot ABSOLUTE"):
        assert spelling in ps_usage.group(0)
        assert spelling in PAGE_TEXT

    for argv, spellings in (
        (["install-artifact", "--help"], ("--version", "--sha256", "--no-init", "--install-root")),
        (["update", "--help"], ("--version",)),
        (["init", "--help"], ("--profile", "--non-interactive", "--backend-url")),
        (["check", "--help"], ()),
        (["--help"], ("--profile", "--config", "--backend-url")),
    ):
        with pytest.raises(SystemExit) as exc:
            proxy_cli.main(argv)
        assert exc.value.code == 0
        actual_help = capsys.readouterr().out
        for spelling in spellings:
            assert spelling in actual_help
            assert spelling in PAGE_TEXT


def test_unmanaged_profile_and_flag_only_examples_normalize_without_launching() -> None:
    commands = {
        line
        for line in PAGE_TEXT.splitlines()
        if line.startswith("forge-proxy init --profile")
        or line.startswith("forge-proxy --backend-url")
    }
    init_commands = [
        line
        for line in commands
        if line.startswith("forge-proxy init") and "--non-interactive" in line
    ]
    launch_commands = [
        line for line in commands if line.startswith("forge-proxy --backend-url")
    ]
    assert len(init_commands) == 2
    assert len(launch_commands) == 2

    for command in init_commands:
        argv = shlex.split(command)[2:]
        args = proxy_cli._build_init_parser().parse_args(argv)
        explicit = {
            name: value
            for name, value in vars(args).items()
            if name not in {"profile", "non_interactive", "force"}
        }
        parsed = _profiles._parse_profile_document(
            {"schema_version": 1, **explicit}
        )
        assert parsed.normalized.backend_url is not None
        if "anthropic-gateway" in command:
            assert parsed.normalized.protocol == "anthropic"

    for command in launch_commands:
        argv = shlex.split(command)[1:]
        parser = proxy_cli._build_parser()
        args = parser.parse_args(argv)
        raw, _verbose, flag_only = proxy_cli._selected_launch(parser, args, argv)
        normalized = _normalize_proxy_config(raw)
        assert flag_only is True
        assert normalized.backend_url is not None
        assert len(supplied_proxy_options(argv)) >= 4


def test_stable_pointer_wording_remains_conditional_for_absent_and_present() -> None:
    prose = " ".join(PAGE_TEXT.split())
    assert "If that pointer is absent" in prose
    assert "no stable standalone Proxy release has been published" in prose
    assert "only while that pointer exists" in prose
    assert "any particular stable or exact standalone release has been published" in prose


def test_custom_root_examples_are_user_owned_absolute_paths() -> None:
    assert "/opt/forge proxy" not in PAGE_TEXT
    assert "C:\\Forge Proxy" not in PAGE_TEXT
    assert PAGE_TEXT.count('"$HOME/.local/share/forge-proxy-custom"') == 4
    assert PAGE_TEXT.count('"$env:LOCALAPPDATA\\Forge Proxy Custom"') == 3


def _heading_anchors(markdown: str) -> set[str]:
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6} +(.*)$", markdown, re.MULTILINE):
        anchor = heading.strip().lower()
        anchor = re.sub(r"[^\w\- ]", "", anchor)
        anchors.add(re.sub(r" +", "-", anchor))
    return anchors


def test_repository_relative_links_and_anchors_resolve() -> None:
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", PAGE_TEXT)
    assert links
    for link in links:
        if re.match(r"^[a-z]+://", link):
            continue
        target_text, _, fragment = link.partition("#")
        target = (PAGE.parent / target_text).resolve()
        assert target.is_file(), link
        assert target.is_relative_to(ROOT), link
        if fragment:
            anchors = _heading_anchors(target.read_text(encoding="utf-8"))
            assert fragment in anchors, link

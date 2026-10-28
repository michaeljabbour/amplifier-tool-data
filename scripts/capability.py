#!/usr/bin/env python3
"""Build, check, install and inspect a self-contained capability release.

Python standard library only. Native tools keep their own install/runtime.
"""

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
E = html.escape


def config():
    data = json.loads((ROOT / "capability.json").read_text())
    for key in ("id", "display_name", "kind", "promise"):
        if not data.get(key):
            raise ValueError("Missing capability field: " + key)
    return data


def within(path, parent):
    value = path.resolve()
    if not value.is_relative_to(parent.resolve()):
        raise ValueError("Path escapes its package: " + str(path))
    return value


def skill_root(data):
    return within(ROOT / data["skill_path"], ROOT) if data.get("skill_path") else None


def portable_files(data):
    folder = skill_root(data)
    if folder is None:
        return []
    if not (folder / "SKILL.md").is_file():
        raise ValueError("Declared portable skill has no entrypoint")
    if folder == ROOT and not data.get("portable_include"):
        raise ValueError("A root-level skill needs an explicit portable_include list")
    candidates = []
    for name in data.get("portable_include", ["."]):
        selected = within(folder / name, folder)
        candidates.extend(selected.rglob("*") if selected.is_dir() else [selected])
    files = []
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ValueError("Portable package contains a symlink: " + str(path))
        if path.is_file() and not any(
            x in {".git", "__pycache__", ".DS_Store"}
            for x in path.relative_to(folder).parts
        ):
            if path.name.startswith(".env"):
                raise ValueError(
                    "Environment files do not belong in a portable release"
                )
            files.append(path)
    entries = [p for p in files if p.name == "SKILL.md"]
    if len(entries) != 1:
        raise ValueError(
            "The portable package must have exactly one canonical SKILL.md"
        )
    for path in files:
        if path.suffix != ".md":
            continue
        for ref in re.findall(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)", path.read_text()):
            uri = urlsplit(ref)
            if uri.scheme or not uri.path:
                continue
            target = within(path.parent / unquote(uri.path), folder)
            if not target.exists():
                raise ValueError("Missing installed reference: " + str(target))
    if not any("LICENSE" in p.name.upper() for p in files):
        raise ValueError("The portable package must carry its license")
    return files


def package_name(data):
    folder = skill_root(data)
    return data.get("skill_name") or (folder.name if folder != ROOT else data["id"])


def safe_url(value):
    value = str(value)
    uri = urlsplit(value)
    return (
        value
        if not value.startswith("//") and uri.scheme in ("", "http", "https")
        else "#"
    )


def inline(text):
    result = E(str(text))
    result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)
    result = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: (
            '<a href="'
            + E(safe_url(html.unescape(m[2])), quote=True)
            + '">'
            + m[1]
            + "</a>"
        ),
        result,
    )
    return result


def markdown(text):
    out = []
    code = []
    in_code = False
    listing = False
    for line in text.splitlines():
        if line.startswith("```"):
            if in_code:
                out.append("<pre><code>" + E("\n".join(code)) + "</code></pre>")
                code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        bullet = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.+)", line)
        if listing and not bullet:
            out.append("</ul>")
            listing = False
        if match:
            level = min(len(match[1]) + 1, 6)
            out.append(f"<h{level}>" + inline(match[2]) + f"</h{level}>")
        elif bullet:
            if not listing:
                out.append("<ul>")
                listing = True
            out.append("<li>" + inline(bullet[1]) + "</li>")
        elif line.strip():
            out.append("<p>" + inline(line) + "</p>")
    if listing:
        out.append("</ul>")
    if code:
        out.append("<pre>" + E("\n".join(code)) + "</pre>")
    return "\n".join(out)


def render_marketing(
    data,
    *,
    download,
    install,
    install_note,
    doc_links,
    method,
    modes,
    example_ui,
    boundaries,
    checks,
    evidence,
    exports,
):
    """Render a product explanation from source-bound copy, without model calls."""
    m = data.get("marketing", {})
    name = E(data["display_name"])
    repo = safe_url(data.get("repository_url", ""))
    source_link = (
        '<a href="' + E(repo, quote=True) + '">Source on GitHub</a>' if repo else ""
    )
    headline = E(m.get("headline", data["promise"]))
    lead = E(m.get("lead", data.get("audience", "")))
    benefits = "".join(
        "<article><h3>" + E(x["title"]) + "</h3><p>" + E(x["body"]) + "</p></article>"
        for x in m.get("benefits", [])
    )
    use_when = "".join(
        "<li>" + E(x) + "</li>"
        for x in m.get("use_when", [data.get("trigger", "")])
        if x
    )
    not_for = "".join("<li>" + E(x) + "</li>" for x in m.get("not_for", []))
    steps = m.get("walkthrough", [])
    step_buttons = "".join(
        '<button type="button" class="walk-step" aria-expanded="'
        + ("true" if i == 0 else "false")
        + '" aria-controls="walk-detail-'
        + str(i)
        + '"><span>'
        + str(i + 1)
        + "</span>"
        + E(x["title"])
        + "</button>"
        for i, x in enumerate(steps)
    )
    step_panels = "".join(
        '<div id="walk-detail-'
        + str(i)
        + '" class="walk-detail"'
        + (" hidden" if i else "")
        + "><h3>"
        + E(x["title"])
        + "</h3><p>"
        + E(x["detail"])
        + "</p></div>"
        for i, x in enumerate(steps)
    )
    walkthrough = (
        '<div class="walkthrough"><div class="walk-controls" role="group" aria-label="Explore the workflow">'
        + step_buttons
        + '</div><div class="walk-panels">'
        + step_panels
        + "</div></div>"
        if steps
        else ""
    )
    prompt_note = (
        "Use after installing in a host that loads skill folders. Supply your own task files and review the result."
        if data.get("skill_path")
        else "Use after installing the documented native package. Read its help, supply an explicit task and inspect the result."
    )
    prompt = m.get("quickstart_prompt", "")
    prompt_box = (
        '<div class="prompt-box"><div class="box-heading"><h3>Try this with your agent</h3><button type="button" class="copy-button" data-copy="starter-prompt">Copy prompt</button></div><pre id="starter-prompt"><code>'
        + E(prompt)
        + '</code></pre><p class="note">'
        + E(prompt_note)
        + "</p></div>"
        if prompt
        else ""
    )
    preview = (
        '<figure class="artifact"><figcaption><span>What you get</span><span>Fictional illustration</span></figcaption><pre>'
        + E(m.get("output_preview", data["promise"]))
        + "</pre><p>"
        + E(
            m.get(
                "example_note",
                "An authored illustration of the output, not a measured user result.",
            )
        )
        + "</p></figure>"
    )
    if repo and repo.startswith("https://github.com/"):
        repo_name = urlsplit(repo).path.rstrip("/").split("/")[-1]
        clone = "git clone " + repo.rstrip("/") + ".git\ncd " + repo_name
        install = (
            '<h3>Start from the public source</h3><div class="code-box"><button type="button" class="copy-button" data-copy="clone-commands">Copy</button><pre id="clone-commands"><code>'
            + E(clone)
            + "</code></pre></div>"
            + install
        )
    install_notes = "".join("<li>" + E(x) + "</li>" for x in m.get("install_notes", []))
    portable_intro = (
        "<p>The download contains the complete skill folder. Extract it into the skills directory supported by your agent, keeping its references and license together. Or use the source installer below. Python 3.11+ is required for that installer.</p>"
        if data.get("skill_path")
        else "<p>This is a native tool or library. Install its package in an isolated environment, then use the documented command or protocol.</p>"
    )
    benefits_section = (
        '<section class="benefits" aria-label="What this helps you do">'
        + benefits
        + "</section>"
        if benefits
        else ""
    )
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f6f4ee"><meta name="description" content="{E(data["promise"], quote=True)}"><meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"><title>{name} | Amplifier</title><link rel="stylesheet" href="style.css"><script src="site.js" defer></script></head>
<body><a class="skip" href="#main">Skip to content</a><header class="masthead"><div class="wrap masthead-inner"><a class="brand" href="#main"><span class="brand-mark" aria-hidden="true"></span><span>Amplifier<small>Skills and tools for thoughtful work</small></span></a><nav aria-label="Main navigation"><a href="#try">Explore</a><a href="#method">How it works</a><a href="#install">Install</a><a href="#evidence">Evidence</a></nav></div></header>
<main id="main" class="wrap"><section class="hero"><div class="hero-copy"><p class="kind">{E(data["kind"].replace("-", " ").capitalize())}</p><h1>{name}</h1><p class="promise">{headline}</p><p class="lede">{lead}</p><div class="actions"><a class="button primary" href="#install">Use {name}</a><a class="text-link" href="#try">Walk through an example</a></div></div>{preview}</section>
{benefits_section}
<section id="try" class="section"><div class="section-heading"><h2>See the work take shape.</h2><p>{E(m.get("example_note", "These examples are fictional. They illustrate the intended workflow and its limits."))}</p></div>{walkthrough}<div class="worked-example">{example_ui}</div></section>
<section id="method" class="section"><div class="section-heading"><h2>A clear job.<br>A useful stopping point.</h2><p>{E(data.get("trigger", ""))}</p></div><div class="fit-grid"><div><h3>Use it when</h3><ul>{use_when}</ul></div><div><h3>Choose another route when</h3><ul>{not_for}</ul></div></div><div class="mode-grid">{modes}</div><details class="method-details"><summary>Read the canonical method</summary><div class="method">{method}</div></details><details><summary>Limits and prerequisites</summary><ul>{boundaries}</ul></details></section>
<section id="install" class="section"><div class="section-heading"><h2>Your first useful run.</h2><p>{E(m.get("next_step", "Install the capability, bring one real task and inspect the result before relying on it."))}</p></div><div class="install-grid"><div>{portable_intro}<div class="actions">{download}{source_link}</div>{install}<p class="note">{install_note}</p><ul class="install-notes">{install_notes}</ul></div><div>{prompt_box}<div class="reading"><h3>Keep the guide nearby</h3><p>{doc_links}</p></div></div></div><p class="copy-status" role="status" aria-live="polite"></p></section>
<section id="evidence" class="section evidence"><div class="section-heading"><h2>Know what you can rely on.</h2><p>{E(m.get("proof_summary", html.unescape(evidence)))}</p></div><details><summary>Read the checks and their scope</summary><ul class="checks">{checks}</ul><p>{evidence}</p></details><p class="note">Fictional examples explain the capability. Tests check the stated behavior. Outside-user outcomes need their own evidence.</p></section></main>
<footer><div class="wrap footer-grid"><div><strong>{name}</strong><p>One capability. Its method, source and limits in reach.</p></div><div>{source_link}<p>{doc_links}</p></div></div><div class="wrap attribution">The family theme is MIT licensed. No institutional affiliation or endorsement is implied.</div></footer></body></html>'''


def build(data):
    output = ROOT / "_site"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    assets = within(ROOT / data.get("site_assets", "site/capability"), ROOT)
    for name in ("style.css", "site.js"):
        shutil.copyfile(assets / name, output / name)
    shutil.copytree(assets / "theme", output / "theme")
    docs = list(
        dict.fromkeys(
            ["README.md", "LICENSE", "CHANGELOG.md", "capability.json"]
            + data.get("docs", [])
        )
    )
    exports = []
    for name in docs:
        source = within(ROOT / name, ROOT)
        if source.is_file():
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            exports.append(name)
    files = portable_files(data)
    folder = skill_root(data)
    download = ""
    if files:
        dest = output / "downloads"
        dest.mkdir()
        archive = dest / (data["id"] + "-skill.zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in files:
                info = zipfile.ZipInfo(
                    package_name(data) + "/" + path.relative_to(folder).as_posix(),
                    (2026, 1, 1, 0, 0, 0),
                )
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                handle.writestr(info, path.read_bytes())
        download = (
            '<a class="button" href="downloads/'
            + E(archive.name)
            + '">Download the portable skill</a>'
        )
        for path in files:
            dest = output / data["skill_path"] / path.relative_to(folder)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    modes = "".join(
        "<article><h3>"
        + E(str(m.get("name", "")))
        + "</h3><p>"
        + E(str(m.get("when", "")))
        + '</p><p class="muted">'
        + E(str(m.get("output", "")))
        + "</p></article>"
        for m in data.get("modes", [])
    )
    examples = data.get("examples", [])
    if any(x.get("fictional") is not True for x in examples):
        raise ValueError("Every illustrative example must be explicitly fictional")
    options = "".join(
        f'<option value="{i}">' + E(str(x.get("title", "Example"))) + "</option>"
        for i, x in enumerate(examples)
    )
    panels = "".join(
        '<article class="example" id="example-'
        + str(i)
        + '"'
        + (" hidden" if i else "")
        + '><p class="eyebrow">Fictional example</p><h3>'
        + E(str(x.get("title", "")))
        + "</h3><p><strong>The brief</strong> "
        + E(str(x.get("brief", "")))
        + '</p><div class="comparison"><div><h4>Before</h4><p>'
        + E(str(x.get("before", "")))
        + "</p></div><div><h4>After applying the method</h4><p>"
        + E(str(x.get("after", "")))
        + "</p></div></div><p>"
        + E(str(x.get("why", "")))
        + "</p></article>"
        for i, x in enumerate(examples)
    )
    example_ui = (
        '<label for="example-select">Choose a fictional task</label><select id="example-select">'
        + options
        + "</select>"
        + panels
        if examples
        else "<p>Use the documented task checks below. No worked outcome is claimed.</p>"
    )
    if folder:
        source = (folder / "SKILL.md").read_text().split("---", 2)[-1]

        def resolve_method_link(match):
            ref = match[2]
            uri = urlsplit(ref)
            if uri.scheme or uri.netloc:
                return match[0]
            base = (
                (folder / uri.path).relative_to(ROOT).as_posix()
                if uri.path
                else (folder / "SKILL.md").relative_to(ROOT).as_posix()
            )
            return (
                "["
                + match[1]
                + "]("
                + base
                + ("#" + uri.fragment if uri.fragment else "")
                + ")"
            )

        source = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", resolve_method_link, source)
        method = markdown(source)
    else:
        method = markdown(
            data.get(
                "method_summary",
                "Use the native tool and its documented command interface.",
            )
        )
    commands = data.get("install_commands", [])
    if len(commands) > 3:
        raise ValueError("The primary install has more than three commands")
    install = (
        "<pre><code>" + E("\n".join(commands)) + "</code></pre>"
        if commands
        else "<p>Read the native installation guide linked below.</p>"
    )
    native = data.get("runtime_install_commands", [])
    if native and (folder or native != commands):
        install += (
            "<details><summary>Native tools and optional execution</summary><pre><code>"
            + E("\n".join(native))
            + "</code></pre><p>These commands install the execution backend. The portable method is a separate surface.</p></details>"
        )
    install_note = (
        "Keep the complete skill folder together, including its references and license. The host applies the method in your context; the skill does not start a background service."
        if folder
        else "Use the native package and command interface described in the installation guide. This capability does not require a portable skill."
    )
    evidence = E(
        data.get(
            "evidence_summary",
            "Source and conformance checks are separate from demonstrated benefit. Comparative effectiveness and outside-user outcomes are not established by this site.",
        )
    )
    checks = "".join(
        "<li><strong>"
        + E(str(v.get("name", "Check")))
        + "</strong> — "
        + E(str(v.get("status", "planned")))
        + "<p>"
        + E(str(v.get("evidence", v.get("expected", ""))))
        + "</p></li>"
        for v in data.get("verification", [])
    )
    doc_links = " · ".join(
        '<a href="'
        + E(name, quote=True)
        + '">'
        + E(Path(name).stem.replace("-", " "))
        + "</a>"
        for name in exports
    )
    boundaries = "".join(
        "<li>" + E(str(x)) + "</li>" for x in data.get("boundaries", [])
    )
    page = render_marketing(
        data,
        download=download,
        install=install,
        install_note=install_note,
        doc_links=doc_links,
        method=method,
        modes=modes,
        example_ui=example_ui,
        boundaries=boundaries,
        checks=checks,
        evidence=evidence,
        exports=exports,
    )
    (output / "index.html").write_text(page)
    receipt = {
        "date": datetime.now(timezone.utc).isoformat(),
        "tool_version": VERSION,
        "kind": "local packaging conformance",
        "portable_files": len(files),
        "capability_sha256": hashlib.sha256(
            (ROOT / "capability.json").read_bytes()
        ).hexdigest(),
        "benefit": "not measured by build",
    }
    (output / "build-record.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "built": str(output),
                "portable_files": len(files),
                "examples": len(examples),
            }
        )
    )


def check(data):
    files = portable_files(data)
    output = ROOT / "_site"
    if not (output / "index.html").exists():
        raise ValueError("Build the site before checking")

    class Page(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
            self.ids = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
            for key in ("href", "src"):
                if key in attrs:
                    self.links.append(attrs[key])

    page = Page()
    page.feed((output / "index.html").read_text())
    if len(page.ids) != len(set(page.ids)):
        raise ValueError("Duplicate page IDs")
    for ref in page.links:
        uri = urlsplit(ref)
        if uri.scheme or uri.netloc:
            continue
        if uri.path and not within(output / unquote(uri.path), output).exists():
            raise ValueError("Broken site link: " + ref)
        if not uri.path and uri.fragment not in page.ids:
            raise ValueError("Missing anchor: " + ref)
    if files:
        folder = skill_root(data)
        with zipfile.ZipFile(
            output / "downloads" / (data["id"] + "-skill.zip")
        ) as archive:
            expected = {
                package_name(data)
                + "/"
                + p.relative_to(folder).as_posix(): p.read_bytes()
                for p in files
            }
            if set(archive.namelist()) != set(expected):
                raise ValueError("ZIP membership drift")
            if any(archive.read(name) != value for name, value in expected.items()):
                raise ValueError("ZIP content drift")
    for name in ("style.css", "theme/style.css"):
        if re.search(
            r"@import|url\(\s*[\"\x27]?(?:https?:|//)", (output / name).read_text()
        ):
            raise ValueError("External runtime asset")
    if not (output / "theme/LICENSE").is_file():
        raise ValueError("Missing theme license")
    print(
        json.dumps(
            {
                "check": "passed",
                "portable_files": len(files),
                "local_links": len(page.links),
                "claim": "Conformance only; not efficacy",
            }
        )
    )


def install(data, destination, replace=False):
    files = portable_files(data)
    source = skill_root(data)
    if source is None:
        raise ValueError("Use this capability's native installation commands")
    parent = destination.expanduser().resolve()
    target = parent / package_name(data)
    if target.is_symlink():
        raise ValueError("Refusing a symlink skill destination")
    identical = target.is_dir() and all(
        (target / p.relative_to(source)).is_file()
        and (target / p.relative_to(source)).read_bytes() == p.read_bytes()
        for p in files
    )
    if target.exists() and not identical:
        if not replace:
            raise ValueError(
                "Existing skill differs. Review it or use --replace to retain a dated backup."
            )
        backup = target.with_name(
            target.name + ".backup-" + datetime.now().strftime("%Y%m%d%H%M%S%f")
        )
        target.rename(backup)
        print("Retained backup: " + str(backup))
    if not identical:
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".capability-install-", dir=parent
        ) as tmp:
            staging = Path(tmp) / package_name(data)
            staging.mkdir()
            for path in files:
                dest = staging / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
            staging.rename(target)
    print(
        json.dumps(
            {"installed": str(target), "files": len(files), "unchanged": identical}
        )
    )


def doctor(data):
    files = portable_files(data)
    print(
        json.dumps(
            {
                "capability": data["id"],
                "portable_method": "valid"
                if files
                else "native tool; no portable method",
                "portable_files": len(files),
                "native_install_commands": data.get("runtime_install_commands", []),
                "runtime_verified_by_doctor": False,
                "explanation": "This checks packaging, not credentials, remote providers, model quality or production behavior.",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "check", "install", "doctor"))
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        data = config()
        if args.operation == "build":
            build(data)
        elif args.operation == "check":
            check(data)
        elif args.operation == "doctor":
            doctor(data)
        else:
            if args.dest is None:
                parser.error(
                    "install needs --dest pointing at the host skills parent directory"
                )
            install(data, args.dest, args.replace)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

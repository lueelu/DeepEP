# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""Build the optional local editor extension as a VSIX with Python stdlib.

Package layout and VSIX metadata follow Microsoft's vsce implementation:
https://github.com/microsoft/vscode-vsce/blob/main/src/package.ts
(toVsixManifest, toContentTypes, ManifestProcessor and LicenseProcessor).
This packages the dependency-free extension only; it does not install or publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile


VISUAL = Path(__file__).resolve().parents[1]
REPOSITORY = VISUAL.parents[3]
VSIX_NS = "http://schemas.microsoft.com/developer/vsx-schema/2011"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
PACKAGE_NAME = "megamoe-train-source-guide"
PAYLOAD_FILES = ("package.json", "extension.cjs", "README.md")


def string_list(package: dict, field: str) -> list[str]:
    values = package.get(field, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"package.json {field} must be an array of strings")
    return values


def validate_package(package: dict) -> None:
    if package.get("name") != PACKAGE_NAME:
        raise ValueError(f"package.json name must be {PACKAGE_NAME}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", package.get("publisher", "")):
        raise ValueError("package.json publisher is missing or invalid")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", package.get("version", "")):
        raise ValueError("package.json version must be a semantic version")
    engine = package.get("engines", {}).get("vscode")
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("package.json engines.vscode is required")
    if package.get("main") not in ("extension.cjs", "./extension.cjs"):
        raise ValueError("The local package includes extension.cjs as its only entry point")
    if package.get("dependencies") or package.get("browser"):
        raise ValueError("This packager supports only the dependency-free desktop extension")
    kinds = string_list(package, "extensionKind")
    if not kinds or any(kind not in {"ui", "workspace"} for kind in kinds):
        raise ValueError("package.json extensionKind must contain ui and/or workspace")


def xml_bytes(element: ET.Element, namespace: str) -> bytes:
    ET.register_namespace("", namespace)
    ET.indent(element, space="  ")
    return ET.tostring(element, encoding="utf-8", xml_declaration=True) + b"\n"


def make_vsix_manifest(package: dict) -> bytes:
    def element(parent: ET.Element, tag: str, text: str | None = None, **attributes: str) -> ET.Element:
        child = ET.SubElement(parent, f"{{{VSIX_NS}}}{tag}", attributes)
        child.text = text
        return child

    root = ET.Element(f"{{{VSIX_NS}}}PackageManifest", {"Version": "2.0.0"})
    metadata = element(root, "Metadata")
    element(
        metadata,
        "Identity",
        Language="en-US",
        Id=package["name"],
        Version=package["version"],
        Publisher=package["publisher"],
    )
    element(metadata, "DisplayName", package.get("displayName", package["name"]))
    description = element(metadata, "Description", package.get("description", ""))
    description.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    element(metadata, "Tags", ",".join(string_list(package, "keywords")))
    element(metadata, "Categories", ",".join(string_list(package, "categories")))
    element(metadata, "GalleryFlags", "Public Preview" if package.get("preview") else "Public")
    properties = element(metadata, "Properties")
    values = {
        "Microsoft.VisualStudio.Code.Engine": package["engines"]["vscode"],
        "Microsoft.VisualStudio.Code.ExtensionDependencies": ",".join(string_list(package, "extensionDependencies")),
        "Microsoft.VisualStudio.Code.ExtensionPack": ",".join(string_list(package, "extensionPack")),
        "Microsoft.VisualStudio.Code.ExtensionKind": ",".join(string_list(package, "extensionKind")),
        "Microsoft.VisualStudio.Code.LocalizedLanguages": "",
        "Microsoft.VisualStudio.Code.EnabledApiProposals": "",
        "Microsoft.VisualStudio.Code.ExecutesCode": "true",
        "Microsoft.VisualStudio.Services.GitHubFlavoredMarkdown": "true",
        "Microsoft.VisualStudio.Services.Content.Pricing": "Free",
    }
    repository = package.get("repository")
    repository_url = repository.get("url") if isinstance(repository, dict) else repository
    if repository_url:
        values["Microsoft.VisualStudio.Services.Links.Source"] = repository_url
    for key, value in values.items():
        element(properties, "Property", Id=key, Value=value)
    element(metadata, "License", "extension/LICENSE")
    installation = element(root, "Installation")
    element(installation, "InstallationTarget", Id="Microsoft.VisualStudio.Code")
    element(root, "Dependencies")
    assets = element(root, "Assets")
    for asset_type, path in (
        ("Microsoft.VisualStudio.Code.Manifest", "extension/package.json"),
        ("Microsoft.VisualStudio.Services.Content.Details", "extension/README.md"),
        ("Microsoft.VisualStudio.Services.Content.License", "extension/LICENSE"),
    ):
        element(assets, "Asset", Type=asset_type, Path=path, Addressable="true")
    return xml_bytes(root, VSIX_NS)


def make_content_types() -> bytes:
    root = ET.Element(f"{{{CONTENT_NS}}}Types")
    for extension, content_type in (
        ("cjs", "application/javascript"),
        ("json", "application/json"),
        ("md", "text/markdown"),
        ("vsixmanifest", "text/xml"),
    ):
        ET.SubElement(root, f"{{{CONTENT_NS}}}Default", {"Extension": extension, "ContentType": content_type})
    ET.SubElement(root, f"{{{CONTENT_NS}}}Override", {"PartName": "/extension/LICENSE", "ContentType": "text/plain"})
    return xml_bytes(root, CONTENT_NS)


def verify_archive(path: Path, package: dict, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != set(members) or archive.testzip() is not None:
            raise ValueError("VSIX members or CRC verification failed")
        if any(archive.read(name) != payload for name, payload in members.items()):
            raise ValueError("VSIX payload differs from its input snapshot")
        packed = json.loads(archive.read("extension/package.json"))
        manifest = ET.fromstring(archive.read("extension.vsixmanifest"))
        identity = manifest.find(f"{{{VSIX_NS}}}Metadata/{{{VSIX_NS}}}Identity")
        if identity is None or any(
            identity.get(xml_name) != package[json_name]
            for xml_name, json_name in (("Id", "name"), ("Version", "version"), ("Publisher", "publisher"))
        ):
            raise ValueError("VSIX Identity does not match package.json")
        if packed != package:
            raise ValueError("Packaged manifest differs from package.json")
        for asset in manifest.findall(f"{{{VSIX_NS}}}Assets/{{{VSIX_NS}}}Asset"):
            if asset.get("Path") not in members:
                raise ValueError("VSIX asset points to a missing package member")
        ET.fromstring(archive.read("[Content_Types].xml"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="store_true", help="Also stage the VSIX in site/downloads")
    args = parser.parse_args()
    extension = VISUAL / "vscode"
    members = {f"extension/{name}": (extension / name).read_bytes() for name in PAYLOAD_FILES}
    package = json.loads(members["extension/package.json"].decode("utf-8-sig"))
    validate_package(package)
    members["extension/LICENSE"] = (REPOSITORY / "LICENSE").read_bytes()
    members["extension.vsixmanifest"] = make_vsix_manifest(package)
    members["[Content_Types].xml"] = make_content_types()
    output = VISUAL / ".artifacts" / f"{PACKAGE_NAME}-{package['version']}.vsix"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".vsix.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, payload in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload, compresslevel=9)
        verify_archive(temporary, package, members)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"PASS: {output}")
    print(f"Extension: {package['publisher']}.{package['name']}@{package['version']}")
    print(f"SHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}")
    if args.site:
        download = VISUAL / "site" / "downloads" / "megamoe-train-source-guide.vsix"
        download.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, download)
        print(f"Website download: {download}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as error:
        print(f"Cannot package local editor extension: {error}", file=sys.stderr)
        raise SystemExit(1) from error

#!/usr/bin/env python3
"""Verify Java publication files, coordinates, MIT text, and Java bytecode."""
from __future__ import annotations

import argparse
import json
import struct
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def check(plugin_dir: Path, version: str, dist: Path | None = None) -> None:
    meta = tomllib.loads((plugin_dir / 'plugin.toml').read_text())
    group, artifact = meta['plugin']['coordinate'].split(':')
    dist = dist or plugin_dir / 'dist'
    base = f'{artifact}-{version}'
    expected = {base + suffix for suffix in ('.jar', '-sources.jar', '-javadoc.jar', '.pom', '.module')}
    actual = {p.name for p in dist.iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError(f'publication files differ: missing={expected-actual}, extra={actual-expected}')
    pom = ET.parse(dist / f'{base}.pom').getroot()
    ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
    for field, value in [('groupId', group), ('artifactId', artifact), ('version', version)]:
        if pom.findtext(f'm:{field}', namespaces=ns) != value:
            raise ValueError(f'POM {field} differs from {value}')
    if pom.findtext('m:licenses/m:license/m:name', namespaces=ns) != 'MIT':
        raise ValueError('POM license must be MIT')
    for dependency_version in pom.findall('.//m:dependency/m:version', namespaces=ns):
        value = dependency_version.text or ''
        if value.endswith('+') or value in {'latest.release', 'latest.integration'}:
            raise ValueError(f'POM contains a Gradle-only dependency version: {value}')
    module = json.loads((dist / f'{base}.module').read_text())
    component = module['component']
    if (component['group'], component['module'], component['version']) != (group, artifact, version):
        raise ValueError('Gradle module coordinate differs')
    license_text = (plugin_dir / 'LICENSE').read_bytes()
    for name in sorted(expected):
        if not name.endswith('.jar'):
            continue
        with zipfile.ZipFile(dist / name) as archive:
            if archive.read('META-INF/LICENSE') != license_text:
                raise ValueError(f'{name}: packaged license differs')
            for entry in archive.namelist():
                if entry.endswith('.class'):
                    magic, _, major = struct.unpack('>IHH', archive.read(entry)[:8])
                    if magic != 0xCAFEBABE or major > 61:
                        raise ValueError(f'{name}: {entry} must target Java 17 or earlier')
    for class_name in meta.get('smoke', {}).get('imports', []):
        with zipfile.ZipFile(dist / f'{base}.jar') as archive:
            if class_name.replace('.', '/') + '.class' not in archive.namelist():
                raise ValueError(f'missing public class {class_name}')
    print(f'OK: {group}:{artifact}:{version}: publication metadata, license and bytecode')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--dist', type=Path)
    args = parser.parse_args()
    check(args.plugin_dir, args.version, args.dist)


if __name__ == '__main__':
    main()

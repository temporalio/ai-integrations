from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import pytest

from check_java_dist import check


def publication(tmp_path: Path) -> Path:
    plugin = tmp_path / 'java' / 'example'
    dist = plugin / 'dist'
    dist.mkdir(parents=True)
    (plugin / 'LICENSE').write_text('MIT license text')
    (plugin / 'plugin.toml').write_text('[plugin]\ncoordinate="io.temporal:example"\n[smoke]\nimports=["io.temporal.example.Public"]\n')
    base = 'example-1.41.0-RC1'
    for suffix in ('.jar', '-sources.jar', '-javadoc.jar'):
        with zipfile.ZipFile(dist / (base + suffix), 'w') as z:
            z.writestr('META-INF/LICENSE', 'MIT license text')
            if suffix == '.jar':
                z.writestr('io/temporal/example/Public.class', struct.pack('>IHH', 0xCAFEBABE, 0, 61))
    (dist / (base + '.pom')).write_text('''<project xmlns="http://maven.apache.org/POM/4.0.0">
<groupId>io.temporal</groupId><artifactId>example</artifactId><version>1.41.0-RC1</version>
<licenses><license><name>MIT</name></license></licenses></project>''')
    (dist / (base + '.module')).write_text(json.dumps({'component': {
        'group': 'io.temporal', 'module': 'example', 'version': '1.41.0-RC1'}}))
    return plugin


def test_valid_publication(tmp_path: Path) -> None:
    check(publication(tmp_path), '1.41.0-RC1')


@pytest.mark.parametrize('version', ['1.1.+', 'latest.release', 'latest.integration'])
def test_publication_rejects_gradle_only_dependency_versions(tmp_path: Path, version: str) -> None:
    plugin = publication(tmp_path)
    pom = plugin / 'dist/example-1.41.0-RC1.pom'
    pom.write_text(pom.read_text().replace('</project>', f'''
<dependencyManagement><dependencies><dependency>
<groupId>org.example</groupId><artifactId>example-bom</artifactId>
<version>{version}</version><type>pom</type><scope>import</scope>
</dependency></dependencies></dependencyManagement></project>'''))
    with pytest.raises(ValueError, match='Gradle-only dependency version'):
        check(plugin, '1.41.0-RC1')


@pytest.mark.parametrize('fault', ['missing', 'extra', 'version', 'license', 'bytecode'])
def test_publication_rejects_invalid_artifacts(tmp_path: Path, fault: str) -> None:
    plugin = publication(tmp_path)
    dist = plugin / 'dist'
    base = 'example-1.41.0-RC1'
    if fault == 'missing':
        (dist / (base + '-sources.jar')).unlink()
    elif fault == 'extra':
        (dist / 'unexpected.jar').write_bytes(b'')
    elif fault == 'version':
        pom = dist / (base + '.pom')
        pom.write_text(pom.read_text().replace('1.41.0-RC1', '0.0.0'))
    else:
        with zipfile.ZipFile(dist / (base + '.jar'), 'w') as z:
            z.writestr('META-INF/LICENSE', 'other' if fault == 'license' else 'MIT license text')
            z.writestr('io/temporal/example/Public.class', struct.pack('>IHH', 0xCAFEBABE, 0, 65))
    with pytest.raises(ValueError):
        check(plugin, '1.41.0-RC1')

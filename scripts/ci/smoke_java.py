#!/usr/bin/env python3
"""Install a Java plugin in a clean Gradle consumer and load its public classes.

Default: create a local Maven repository from the tested distributions. A release
can pass an authenticated repository URL to prove staged or public installation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


def relocation_source(pom: Path, coordinate: str, version: str) -> tuple[str, str, str]:
    """Validate a relocation POM and return its original Maven coordinates."""
    root = ET.parse(pom).getroot()
    ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
    group, artifact = coordinate.split(':')
    target = root.find('m:distributionManagement/m:relocation', ns)
    if target is None or root.findtext('m:packaging', namespaces=ns) != 'pom':
        raise ValueError('expected a POM-only Maven relocation')
    for field, expected in [('groupId', group), ('artifactId', artifact), ('version', version)]:
        if target.findtext('m:' + field, namespaces=ns) != expected:
            raise ValueError(f'relocation target {field} differs from {expected}')
    source = tuple(root.findtext('m:' + field, namespaces=ns) or ''
                   for field in ('groupId', 'artifactId', 'version'))
    if not all(source) or source == (group, artifact, version):
        raise ValueError('relocation must specify distinct original coordinates')
    if root.find('m:dependencies', ns) is not None:
        raise ValueError('relocation POM must not declare dependencies')
    return source


def smoke(plugin_dir: Path, version: str, repository: str | None = None,
          relocation_pom: Path | None = None) -> None:
    plugin_dir = plugin_dir.resolve()
    meta = tomllib.loads((plugin_dir / 'plugin.toml').read_text())
    group, artifact = meta['plugin']['coordinate'].split(':')
    source = relocation_source(relocation_pom, f'{group}:{artifact}', version) if relocation_pom else None
    modules = [(group, artifact)] + ([source[:2]] if source else [])
    with tempfile.TemporaryDirectory(prefix='java-consumer-') as directory:
        consumer = Path(directory)
        if repository is None:
            repo = consumer / 'repository' / group.replace('.', '/') / artifact / version
            repo.mkdir(parents=True)
            for path in (plugin_dir / 'dist').iterdir():
                if path.suffix in {'.jar', '.pom', '.module'}:
                    shutil.copyfile(path, repo / path.name)
            if source:
                old_group, old_artifact, old_version = source
                old_repo = consumer / 'repository' / old_group.replace('.', '/') / old_artifact / old_version
                old_repo.mkdir(parents=True)
                shutil.copyfile(relocation_pom, old_repo / f'{old_artifact}-{old_version}.pom')
            repository = (consumer / 'repository').as_uri()
        dependency = ':'.join(source) if source else f'{group}:{artifact}:{version}'
        dependencies = [dependency, *meta.get('smoke', {}).get('dependencies', [])]
        quoted_deps = '\n'.join('    implementation ' + json.dumps(dep) for dep in dependencies)
        includes = '; '.join(f'includeModule({json.dumps(g)}, {json.dumps(a)})' for g, a in modules)
        excludes = '; '.join(f'excludeModule({json.dumps(g)}, {json.dumps(a)})' for g, a in modules)
        (consumer / 'settings.gradle').write_text("rootProject.name = 'isolated-consumer'\n")
        # Credentials stay in environment variables, not generated files or CLI arguments.
        (consumer / 'build.gradle').write_text('''plugins { id 'application' }
repositories {
    maven {
        url = uri(''' + json.dumps(repository) + ''')
        content { ''' + includes + ''' }
        if (System.getenv('CENTRAL_BEARER')) {
            credentials(HttpHeaderCredentials) {
                name = 'Authorization'
                value = 'Bearer ' + System.getenv('CENTRAL_BEARER')
            }
            authentication { header(HttpHeaderAuthentication) }
        }
    }
    mavenCentral {
        content { ''' + excludes + ''' }
    }
}
dependencies {
''' + quoted_deps + '''
}
application { mainClass = 'Consumer' }
''')
        src = consumer / 'src/main/java'
        src.mkdir(parents=True)
        classes = ', '.join(json.dumps(c) for c in meta['smoke']['imports'])
        (src / 'Consumer.java').write_text('''public class Consumer {
    public static void main(String[] args) throws Exception {
        for (String name : new String[] {''' + classes + '''}) {
            Class<?> type = Class.forName(name);
            String location = type.getProtectionDomain().getCodeSource().getLocation().toString();
            if (!location.endsWith("''' + artifact + '-' + version + '''.jar")) {
                throw new AssertionError("Unexpected plugin origin: " + location);
            }
            System.out.println("Loaded " + name);
        }
    }
}
''')
        command = ['bash', str(plugin_dir / 'gradlew'), '--no-daemon', '-p', str(consumer), 'run']
        subprocess.run(command, cwd=plugin_dir, check=True, env=os.environ.copy())
    print(f'OK: clean consumer installed {group}:{artifact}:{version}')
    if source:
        print(f'OK: {":".join(source)} relocated to {group}:{artifact}:{version}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--repository')
    parser.add_argument('--relocation-pom', type=Path, help='resolve through this POM\'s original coordinate')
    args = parser.parse_args()
    smoke(args.plugin_dir, args.version, args.repository, args.relocation_pom)

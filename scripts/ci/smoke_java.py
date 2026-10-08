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
from pathlib import Path


def smoke(plugin_dir: Path, version: str, repository: str | None = None) -> None:
    plugin_dir = plugin_dir.resolve()
    meta = tomllib.loads((plugin_dir / 'plugin.toml').read_text())
    group, artifact = meta['plugin']['coordinate'].split(':')
    with tempfile.TemporaryDirectory(prefix='java-consumer-') as directory:
        consumer = Path(directory)
        if repository is None:
            repo = consumer / 'repository' / group.replace('.', '/') / artifact / version
            repo.mkdir(parents=True)
            for path in (plugin_dir / 'dist').iterdir():
                if path.suffix in {'.jar', '.pom', '.module'}:
                    shutil.copyfile(path, repo / path.name)
            repository = (consumer / 'repository').as_uri()
        dependencies = [f'{group}:{artifact}:{version}', *meta.get('smoke', {}).get('dependencies', [])]
        quoted_deps = '\n'.join('    implementation ' + json.dumps(dep) for dep in dependencies)
        (consumer / 'settings.gradle').write_text("rootProject.name = 'isolated-consumer'\n")
        # Credentials stay in environment variables, not generated files or CLI arguments.
        (consumer / 'build.gradle').write_text('''plugins { id 'application' }
repositories {
    maven {
        url = uri(''' + json.dumps(repository) + ''')
        content { includeModule(''' + json.dumps(group) + ', ' + json.dumps(artifact) + ''') }
        if (System.getenv('CENTRAL_BEARER')) {
            credentials(HttpHeaderCredentials) {
                name = 'Authorization'
                value = 'Bearer ' + System.getenv('CENTRAL_BEARER')
            }
            authentication { header(HttpHeaderAuthentication) }
        }
    }
    mavenCentral {
        content { excludeModule(''' + json.dumps(group) + ', ' + json.dumps(artifact) + ''') }
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--repository')
    args = parser.parse_args()
    smoke(args.plugin_dir, args.version, args.repository)

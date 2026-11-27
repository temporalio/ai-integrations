// Reuse the same CLI flags as the SDK's local test environment helper.
const { TypedSearchAttributes } = require('@temporalio/common');
const { defaultDynamicConfigOptions, defaultSAKeys } = require('./helpers/lib/environment');

process.stdout.write(JSON.stringify([
  ...defaultDynamicConfigOptions.flatMap((option) => ['--dynamic-config-value', option]),
  ...Object.values(defaultSAKeys).flatMap(({ name, type }) => [
    '--search-attribute', `${name}=${TypedSearchAttributes.toMetadataType(type)}`,
  ]),
]));

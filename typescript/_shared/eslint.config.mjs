// Upstream SDK rules, resolved from the independently installed plugin.
import { createRequire } from 'node:module';
const require = createRequire(`${process.cwd()}/package.json`);
const js = require('@eslint/js');
const tseslint = require('typescript-eslint');
const importPlugin = require('eslint-plugin-import');
const { defaultConditionNames } = require('eslint-import-resolver-typescript');
const prettierConfig = require('eslint-config-prettier');

export default tseslint.config(
  { ignores: ['**/node_modules/**', '**/lib/**', '**/*.js', '**/*.mjs', '**/*.cjs', '**/*.generated.ts'] },
  {
    files: ['src/**/*.ts'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended, prettierConfig],
    plugins: { import: importPlugin },
    languageOptions: {
      parserOptions: { project: ['./tsconfig.json'], tsconfigRootDir: process.cwd() },
    },
    settings: {
      'import/parsers': { '@typescript-eslint/parser': ['.ts'] },
      'import/resolver': {
        // Resolve types under `<root>@types` even for packages without source code, like `@types/unist`
        typescript: {
          alwaysTryTypes: true,
          project: ['./tsconfig.json'],
        },
      },
      'import/internal-regex': '^@temporalio/',
    },
    rules: {
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'no-duplicate-imports': ['error', { allowSeparateTypeImports: true }],
      'object-shorthand': ['error', 'always'],
      'no-restricted-imports': ['error', { patterns: ['@temporalio/*/src/*'] }],
      '@typescript-eslint/no-deprecated': [
        'warn',
        {
          allow: [
            // Untyped Search Attributes
            {
              from: 'file',
              name: [
                'SearchAttributes',
                'searchAttributes',
                'searchAttributeValue',
                'SearchAttributeValue',
                'SearchAttributeValueOrReadonly',
              ],
            },

            // Worker Versioning v2
            {
              from: 'file',
              name: [
                'AddNewCompatibleVersion',
                'AddNewIdInNewDefaultSet',
                'BaseReachabilityOptions',
                'buildId',
                'BuildIdNotFoundError',
                'BuildIdOperation',
                'BuildIdReachability',
                'BuildIdVersionSet',
                'currentBuildId',
                'getBuildIdCompatability',
                'getReachability',
                'LoadedTaskQueueClientOptions',
                'MergeSets',
                'PromoteBuildIdWithinSet',
                'PromoteSetByBuildId',
                'ReachabilityOptions',
                'ReachabilityResponse',
                'ReachabilityType',
                'ReachabilityTypeResponse',
                'TaskQueueClient',
                'TaskQueueClientOptions',
                'UnversionedBuildId',
                'updateBuildIdCompatibility',
                'useVersioning',
                'versioningIntent',
                'VersioningIntent',
                'versioningIntentToProto',
                'VersioningIntentString',
                'workerBuildId',
                'WorkerBuildIdVersionSets',
              ],
            },

            // Activity Inbound Calls Interceptors and Workflow Client Calls Interceptor
            {
              from: 'file',
              name: [
                'activityInbound',
                'ActivityInboundCallsInterceptorFactory',
                'WorkflowClientInterceptors',
                'WorkflowClientCallsInterceptor',
                'WorkflowClientCallsInterceptorFactory',
                'WorkflowClientCallsInterceptorFactoryInput',
              ],
            },

            // Legacy Log Interceptors / Sinks
            {
              from: 'file',
              name: [
                'ActivityInboundLogInterceptor',
                'defaultSinks',
                'defaultWorkerLogger',
                'LoggerSinks',
                'WorkflowLogInterceptor',
              ],
            },
          ],
        },
      ],
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/explicit-module-boundary-types': 'error',
      '@typescript-eslint/consistent-type-imports': [
        'error',
        {
          prefer: 'type-imports',
          fixStyle: 'separate-type-imports',
          disallowTypeAnnotations: false,
        },
      ],
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      // Import rules
      'import/no-unresolved': ['error', { ignore: ['^__temporal_'] }],
      // TypeScript compilation already ensures that named imports exist in the referenced module
      'import/named': 'off',
      'import/default': 'error',
      'import/namespace': 'error',
      'import/no-absolute-path': 'error',
      'import/no-self-import': 'error',
      'import/no-cycle': 'error',
      'import/no-useless-path-segments': 'error',
      'import/no-relative-packages': 'error',
      'import/export': 'error',
      'import/no-named-as-default': 'error',
      'import/no-extraneous-dependencies': 'error',
      'import/no-mutable-exports': 'error',
      'import/unambiguous': 'error',
      'import/first': 'error',
      'import/order': ['error', { groups: ['builtin', 'external', 'internal', 'parent', 'sibling', 'index'] }],
      'import/newline-after-import': 'error',
      'import/no-unassigned-import': 'error',
      'import/no-named-default': 'error',
    },
  },
  {
    files: ['src/__tests__/**/*.ts'],
    settings: {
      'import/resolver': {
        typescript: {
          // MCP's wildcard types export maps `.js` imports to nonexistent `.js.d.ts` files.
          // Resolve runtime exports instead; extension aliases still find their declarations.
          conditionNames: defaultConditionNames.filter((condition) => condition !== 'types'),
        },
      },
    },
  },
  {
    files: ['src/**/*.ts'],
    ignores: ['src/__tests__/**'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: ['@temporalio/*/src/*', '**/__tests__/**', '**/*.test'],
        },
      ],
    },
  }
);

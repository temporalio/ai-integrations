// Check every resolved SDK package, including a dependency's separate copy.
exports.requireRuntimeHooks = {
  name: 'RequireAgentsRuntimeHooks',
  configureBundler(options) {
    const previous = options.webpackConfigHook;
    return { ...options, webpackConfigHook(config) {
      const configured = previous ? previous(config) : config;
      const check = compiler => compiler.hooks.normalModuleFactory.tap('RequireAgentsRuntimeHooks', factory => {
        factory.hooks.afterResolve.tap('RequireAgentsRuntimeHooks', data => {
          const resolved = data.createData?.resourceResolveData;
          const pkg = resolved?.descriptionFileData;
          if (pkg?.name === '@openai/agents-core' && !pkg.exports?.['./runtime-policy']) {
            throw new Error(`Unsupported agents-core runtime: ${resolved.descriptionFilePath}`);
          }
        });
      });
      return { ...configured, plugins: [...(configured.plugins || []), check] };
    } };
  },
};

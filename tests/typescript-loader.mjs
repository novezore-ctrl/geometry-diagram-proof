const TYPESCRIPT_SUFFIXES = [".ts", ".tsx"];

export async function resolve(specifier, context, nextResolve) {
  try {
    return await nextResolve(specifier, context);
  } catch (error) {
    const isRelativeWithoutExtension = specifier.startsWith(".") && !/\.[cm]?[jt]sx?$/.test(specifier);
    if (error?.code !== "ERR_MODULE_NOT_FOUND" || !isRelativeWithoutExtension) throw error;

    for (const suffix of TYPESCRIPT_SUFFIXES) {
      try {
        return await nextResolve(`${specifier}${suffix}`, context);
      } catch (candidateError) {
        if (candidateError?.code !== "ERR_MODULE_NOT_FOUND") throw candidateError;
      }
    }
    throw error;
  }
}

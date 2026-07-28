import { CJS_ENUM_NAME, CjsSchema } from "./CjsSchema.js";

// Namespace decorators re-exported as named bindings so consumers can write
// `import { type, io } from ".../schema"` and `@type.string` instead of `@CjsSchema.type.string`.
const { type, io, jessica, impl, carbon, components } = CjsSchema;

export {
    carbon,
    components,
    CJS_ENUM_NAME,
    CjsSchema,
    CjsSchema as schema,
    impl,
    io,
    jessica,
    type
};

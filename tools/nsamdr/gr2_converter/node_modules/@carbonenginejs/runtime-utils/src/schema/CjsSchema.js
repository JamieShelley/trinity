const CLASS_SCHEMA = new WeakMap();
const CONSTRUCTOR_BY_NAME = new Map();
const ENUM_SCHEMA_BY_NAME = new Map();
const ENUM_SCHEMA_BY_OBJECT = new WeakMap();
const STAGE3_FIELD_METADATA = Symbol("carbonenginejs.schema.stage3Fields");

export const CJS_ENUM_NAME = Symbol.for("carbonenginejs.enum.name");

/**
 * Reusable schema/decorator metadata surface.
 *
 * Decorators are namespace-scoped so consumers can export only the parts they
 * understand. The functions support stage-3 field decorators and direct tool
 * registration through decorateField().
 */
export class CjsSchema
{
    static define(Constructor, definition = {})
    {
        defineClassMetadata(Constructor, normalizeClassDefinition(Constructor, definition));
        return this;
    }

    static enum(values)
    {
        return fieldDecorator("enum", normalizeEnumDefinition(values));
    }

    static defineEnum(values, definition = {})
    {
        defineEnumMetadata(values, normalizeEnumSchema(values, definition));
        return this;
    }

    static decorateField(Constructor, fieldName, ...decorators)
    {
        for (const decorator of decorators)
        {
            decorator(Constructor.prototype, fieldName);
        }
        return Constructor;
    }

    static decorateMethod(Constructor, methodName, ...decorators)
    {
        for (const decorator of decorators)
        {
            decorator(Constructor.prototype, methodName);
        }
        return Constructor;
    }

    static defineField(Constructor, fieldName, namespace, value)
    {
        defineFieldMetadata(Constructor, fieldName, namespace, value);
        return this;
    }

    static defineMethod(Constructor, methodName, namespace, value)
    {
        defineMethodMetadata(Constructor, methodName, namespace, value);
        return this;
    }

    static getField(Constructor, fieldName)
    {
        return getEffectiveFields(Constructor).find(field => field.name === fieldName) || null;
    }

    /**
     * Excludes named inherited fields from the decorated class's schema surface.
     */
    static hideInherited(fieldNames)
    {
        return hiddenInheritedFieldsDecorator(normalizeHiddenInheritedFields(fieldNames));
    }

    /**
     * Checks whether a field is hidden from a class by its inheritance chain.
     */
    static isFieldHidden(Constructor, fieldName)
    {
        return getHiddenInheritedFieldNames(Constructor).has(fieldName);
    }

    static getMethod(Constructor, methodName)
    {
        const schema = CLASS_SCHEMA.get(Constructor);
        return schema?.methodsByName.get(methodName) || null;
    }

    static getClassName(Constructor)
    {
        let current = Constructor;
        while (typeof current === "function")
        {
            const className = CLASS_SCHEMA.get(current)?.className || null;
            if (className)
            {
                return current !== Constructor && className === "CjsModel" ? null : className;
            }
            current = Object.getPrototypeOf(current);
        }
        return null;
    }

    static getClassFamily(Constructor)
    {
        let current = Constructor;
        while (typeof current === "function")
        {
            const family = CLASS_SCHEMA.get(current)?.family || null;
            if (family) return family;
            current = Object.getPrototypeOf(current);
        }
        return null;
    }

    static SetConstructor(name, Constructor)
    {
        if (typeof name !== "string" || !name.trim())
        {
            throw new TypeError("CjsSchema.SetConstructor requires a non-empty name.");
        }
        if (typeof Constructor !== "function")
        {
            throw new TypeError(`CjsSchema constructor ${name.trim()} must be a function.`);
        }

        CONSTRUCTOR_BY_NAME.set(name.trim(), Constructor);
        return this;
    }

    static GetConstructor(name)
    {
        if (typeof name !== "string" || !name.trim()) return null;
        return CONSTRUCTOR_BY_NAME.get(name.trim()) || null;
    }

    static getEnumName(values)
    {
        if (typeof values === "string") return values;
        return values && typeof values === "object"
            ? ENUM_SCHEMA_BY_OBJECT.get(values)?.name || values[CJS_ENUM_NAME] || values.Source?.name || values.name || null
            : null;
    }

    static getEnum(values)
    {
        const name = CjsSchema.getEnumName(values);
        return name ? ENUM_SCHEMA_BY_NAME.get(name) || null : null;
    }

    static getSchema(Constructor, options = {})
    {
        const schema = CLASS_SCHEMA.get(Constructor);
        const namespaces = normalizeNamespaces(options.namespaces);
        const fields = [];
        const methods = [];

        for (const field of getEffectiveFields(Constructor))
        {
            fields.push(enrichEnumField(exportField(field, namespaces), Constructor));
        }

        for (const method of schema?.methods || [])
        {
            methods.push(exportField(method, namespaces));
        }

        const result = {
            className: CjsSchema.getClassName(Constructor),
            fields: Object.freeze(fields)
        };

        const family = schema?.family || CjsSchema.getClassFamily(Constructor);
        if (family)
        {
            result.family = family;
        }

        if (schema?.sourceClass && schema.sourceClass !== result.className)
        {
            result.sourceClass = schema.sourceClass;
        }

        if (schema?.aliases?.length)
        {
            result.aliases = Object.freeze([...schema.aliases]);
        }

        if (methods.length) result.methods = Object.freeze(methods);

        return Object.freeze(result);
    }

    static type = Object.freeze({
        array: itemType => fieldDecorator("type", { kind: "array", itemType }),
        boolean: fieldDecorator("type", { kind: "boolean" }),
        color: fieldDecorator("type", { kind: "color" }),
        define: definition => classDefinitionDecorator(definition),
        expression: fieldDecorator("type", { kind: "expression", js: "string" }),
        float32: fieldDecorator("type", { kind: "float32" }),
        float64: fieldDecorator("type", { kind: "float64" }),
        int8: fieldDecorator("type", { kind: "int8" }),
        int16: fieldDecorator("type", { kind: "int16" }),
        int32: fieldDecorator("type", { kind: "int32" }),
        int64: fieldDecorator("type", { kind: "int64" }),
        list: itemType => fieldDecorator("type", { kind: "list", itemType }),
        mat3: fieldDecorator("type", { kind: "mat3" }),
        mat4: fieldDecorator("type", { kind: "mat4" }),
        map: valueType => fieldDecorator("type", { kind: "map", valueType }),
        model: className => fieldDecorator("type", { kind: "model", className }),
        objectRef: className => fieldDecorator("type", { kind: "objectRef", className }),
        path: fieldDecorator("type", { kind: "path" }),
        quat: fieldDecorator("type", { kind: "quat" }),
        rawStruct: className => fieldDecorator("type", { kind: "rawStruct", className }),
        set: itemType => fieldDecorator("type", { kind: "set", itemType }),
        string: fieldDecorator("type", { kind: "string" }),
        struct: className => fieldDecorator("type", { kind: "struct", className }),
        typedArray: arrayType => fieldDecorator("type", { kind: "typedArray", arrayType }),
        uint8: fieldDecorator("type", { kind: "uint8" }),
        uint16: fieldDecorator("type", { kind: "uint16" }),
        uint32: fieldDecorator("type", { kind: "uint32" }),
        uint64: fieldDecorator("type", { kind: "uint64" }),
        unknown: fieldDecorator("type", { kind: "unknown" }),
        vec2: fieldDecorator("type", { kind: "vec2" }),
        vec3: fieldDecorator("type", { kind: "vec3" }),
        vec4: fieldDecorator("type", { kind: "vec4" })
    });

    static io = Object.freeze({
        always: fieldDecorator("io", { always: true }),
        notify: fieldDecorator("io", { notify: true }),
        owned: fieldDecorator("io", { ownership: "owned" }),
        persist: fieldDecorator("io", { read: true, write: true, persist: true }),
        persistOnly: fieldDecorator("io", { persist: true, persistOnly: true }),
        read: fieldDecorator("io", { read: true }),
        readwrite: fieldDecorator("io", { read: true, write: true }),
        reference: fieldDecorator("io", { ownership: "reference" }),
        // Declares the lazy-invalidation token(s) a change to this field
        // implies ("bounds is stale"). Added to __state.flags at write time;
        // cleared ONLY by the getter that recomputes the derived value.
        flag: (...tokens) => fieldDecorator("io", { flag: tokens.flat().map(String) }),
        // Declares the rebuild requirement token(s) a change to this field
        // implies ("vertices need rebuilding"). Added to __state.rebuild at
        // write time; cleared ONLY by the specific work method that succeeds
        // (typically driven from Update / per-frame passes).
        rebuild: (...tokens) => fieldDecorator("io", { rebuild: tokens.flat().map(String) }),
        write: fieldDecorator("io", { write: true })
    });

    static jessica = Object.freeze({
        group: name => fieldDecorator("jessica", { group: name }),
        hidden: fieldDecorator("jessica", { hidden: true }),
        readOnly: fieldDecorator("jessica", { readOnly: true }),
        widget: name => fieldDecorator("jessica", { widget: name })
    });

    // impl decorators apply to methods AND fields: a promoted/diverging field
    // (e.g. a Carbon-hidden authored value exposed for values interchange) is
    // an implementation decision, so it carries impl.adapted/impl.custom +
    // impl.reason just like a diverging method. carbon.* stays factual
    // provenance and remains method-only.
    static impl = Object.freeze({
        abstract: memberDecorator("impl", { abstract: true, status: "abstract" }),
        adapted: memberDecorator("impl", { adapted: true, status: "adapted" }),
        custom: memberDecorator("impl", { custom: true, status: "custom" }),
        implemented: memberDecorator("impl", { implemented: true, status: "implemented" }),
        noop: memberDecorator("impl", { noop: true, status: "noop" }),
        notImplemented: memberDecorator("impl", { notImplemented: true, status: "notImplemented" }),
        notSupported: memberDecorator("impl", { notSupported: true, status: "notSupported" }),
        note: text => memberDecorator("impl", { note: String(text) }),
        reason: text => memberDecorator("impl", { reason: String(text) })
    });

    static carbon = Object.freeze({
        method: methodDecorator("carbon", { method: true }),
        renamed: originalName => {
            if (typeof originalName !== "string" || !originalName.trim())
            {
                throw new TypeError("CjsSchema.carbon.renamed requires a non-empty original method name.");
            }
            return methodDecorator("carbon", {
                method: true,
                renamed: true,
                originalName: originalName.trim()
            });
        },
        contextual: tiers => {
            const list = Array.isArray(tiers) ? tiers : [tiers];
            const normalized = [];
            for (const tier of list)
            {
                if (typeof tier !== "string" || !tier.trim())
                {
                    continue;
                }
                normalized.push(tier.trim());
            }
            if (!normalized.length)
            {
                throw new TypeError("CjsSchema.carbon.contextual requires at least one context tier name.");
            }
            const base = methodDecorator("carbon", {
                method: true,
                contextual: true,
                contextTiers: Object.freeze(normalized)
            });
            return function contextualMethodDecorator(targetOrValue, contextOrMethodName)
            {
                // Contextual methods are validated context-first at decoration
                // time: the first declared parameter must be the frame context.
                if (contextOrMethodName && typeof contextOrMethodName === "object")
                {
                    assertContextFirstMethod(targetOrValue, contextOrMethodName.name);
                }
                else if (targetOrValue && contextOrMethodName)
                {
                    assertContextFirstMethod(targetOrValue[contextOrMethodName], contextOrMethodName);
                }
                return base(targetOrValue, contextOrMethodName);
            };
        }
    });

    static components = createComponentsNamespace();
}

function createComponentsNamespace()
{
    const components = definition => fieldDecorator("components", normalizeComponentDefinition(definition));
    Object.defineProperties(components, {
        get: { value: getComponentValue },
        indices: { value: getComponentIndices },
        set: { value: setComponentValue }
    });
    return Object.freeze(components);
}

function fieldDecorator(namespace, value)
{
    return function schemaFieldDecorator(targetOrValue, contextOrFieldName)
    {
        if (contextOrFieldName && typeof contextOrFieldName === "object")
        {
            const context = contextOrFieldName;
            if (context.kind !== "field") throw new TypeError("CjsSchema decorators only support class fields.");
            recordStage3FieldMetadata(context, namespace, value);

            // Register field metadata on instance construction. addInitializer covers
            // spec-compliant runtimes; the returned field initializer covers runtimes (e.g.
            // Deno/SWC) that do NOT fire field-decorator addInitializer. Both register the same
            // metadata (idempotent via mergeNamespace), so whichever the runtime honours, the
            // schema is populated. Registration is lazy (first construction); the class decorator
            // still registers the class eagerly at definition time.
            context.addInitializer(function initializeSchemaField()
            {
                defineFieldMetadata(this.constructor, context.name, namespace, value);
            });

            return function initializeSchemaFieldValue(initialValue)
            {
                defineFieldMetadata(this.constructor, context.name, namespace, value);
                return initialValue;
            };
        }

        const Constructor = targetOrValue?.constructor;
        if (!Constructor || !contextOrFieldName)
        {
            throw new TypeError("CjsSchema field decorators require a class field target.");
        }

        defineFieldMetadata(Constructor, contextOrFieldName, namespace, value);
    };
}

function classDefinitionDecorator(definition)
{
    return function schemaClassDefinitionDecorator(value, context)
    {
        if (context && typeof context === "object")
        {
            if (context.kind !== "class") throw new TypeError("CjsSchema type.define only supports classes.");
            registerStage3FieldMetadata(value, context.metadata);
            defineClassMetadata(value, normalizeClassDefinition(value, definition));
            return;
        }

        if (typeof value !== "function")
        {
            throw new TypeError("CjsSchema type.define requires a class constructor.");
        }

        defineClassMetadata(value, normalizeClassDefinition(value, definition));
    };
}

function hiddenInheritedFieldsDecorator(fieldNames)
{
    return function schemaHiddenInheritedFieldsDecorator(value, context)
    {
        if (context && typeof context === "object")
        {
            if (context.kind !== "class") throw new TypeError("CjsSchema.hideInherited only supports classes.");
            registerStage3FieldMetadata(value, context.metadata);
        }
        else if (typeof value !== "function")
        {
            throw new TypeError("CjsSchema.hideInherited requires a class constructor.");
        }

        defineHiddenInheritedFields(value, fieldNames);
    };
}

const CONTEXT_FIRST_PARAMETER = /^\(?\s*_?(context|updateContext)\b/;

function assertContextFirstMethod(fn, methodName)
{
    if (typeof fn !== "function")
    {
        return;
    }
    const source = String(fn);
    const parameterList = source.slice(source.indexOf("("));
    if (fn.length < 1 || !CONTEXT_FIRST_PARAMETER.test(parameterList))
    {
        throw new TypeError(
            `CjsSchema.carbon.contextual method "${String(methodName)}" must be context-first ` +
            "(first parameter named context or updateContext)."
        );
    }
}

function methodDecorator(namespace, value)
{
    return function schemaMethodDecorator(targetOrValue, contextOrMethodName)
    {
        if (contextOrMethodName && typeof contextOrMethodName === "object")
        {
            const context = contextOrMethodName;
            if (context.kind !== "method") throw new TypeError("CjsSchema method decorators only support class methods.");

            context.addInitializer(function initializeSchemaMethod()
            {
                const Constructor = context.static ? this : this.constructor;
                defineMethodMetadata(Constructor, context.name, namespace, value);
            });
            return;
        }

        const Constructor = targetOrValue?.constructor;
        if (!Constructor || !contextOrMethodName)
        {
            throw new TypeError("CjsSchema method decorators require a class method target.");
        }

        defineMethodMetadata(Constructor, contextOrMethodName, namespace, value);
    };
}

function memberDecorator(namespace, value)
{
    const forMethods = methodDecorator(namespace, value);
    const forFields = fieldDecorator(namespace, value);
    return function schemaMemberDecorator(targetOrValue, contextOrMemberName)
    {
        if (contextOrMemberName && typeof contextOrMemberName === "object")
        {
            return contextOrMemberName.kind === "field"
                ? forFields(targetOrValue, contextOrMemberName)
                : forMethods(targetOrValue, contextOrMemberName);
        }

        // Legacy (non-2023-11) path: a method target resolves to a function
        // on the prototype; anything else is treated as a field.
        return targetOrValue && contextOrMemberName && typeof targetOrValue[contextOrMemberName] === "function"
            ? forMethods(targetOrValue, contextOrMemberName)
            : forFields(targetOrValue, contextOrMemberName);
    };
}

function defineFieldMetadata(Constructor, fieldName, namespace, value)
{
    defineMemberMetadata(Constructor, "fields", "fieldsByName", fieldName, namespace, value);
}

function defineClassMetadata(Constructor, definition)
{
    const schema = getOrCreateClassSchema(Constructor);
    if (definition.className) schema.className = definition.className;
    if (definition.family) schema.family = definition.family;
    if (definition.sourceClass) schema.sourceClass = definition.sourceClass;
    if (definition.aliases) schema.aliases = Object.freeze([...definition.aliases]);

    for (const field of definition.fields || [])
    {
        defineManualMemberMetadata(Constructor, "fields", field);
    }

    for (const method of definition.methods || [])
    {
        defineManualMemberMetadata(Constructor, "methods", method);
    }

    registerClassMetadata(Constructor, schema);
}

function defineManualMemberMetadata(Constructor, memberType, definition)
{
    const define = memberType === "methods" ? defineMethodMetadata : defineFieldMetadata;
    for (const [namespace, value] of Object.entries(definition))
    {
        if (namespace === "name") continue;
        define(Constructor, definition.name, namespace, value);
    }
}

function defineMethodMetadata(Constructor, methodName, namespace, value)
{
    defineMemberMetadata(Constructor, "methods", "methodsByName", methodName, namespace, value);
}

function defineMemberMetadata(Constructor, listKey, mapKey, name, namespace, value)
{
    const schema = getOrCreateClassSchema(Constructor);
    let item = schema[mapKey].get(name);

    if (!item)
    {
        item = { name };
        schema[listKey].push(item);
        schema[mapKey].set(name, item);
    }

    item[namespace] = mergeNamespace(item[namespace], value);
}

function defineHiddenInheritedFields(Constructor, fieldNames)
{
    if (typeof Constructor !== "function")
    {
        throw new TypeError("CjsSchema.hideInherited requires a class constructor.");
    }

    const Parent = Object.getPrototypeOf(Constructor);
    const inheritedFields = new Set(getEffectiveFields(Parent).map(field => field.name));
    const className = CLASS_SCHEMA.get(Constructor)?.className || Constructor.name || "<anonymous>";

    for (const fieldName of fieldNames)
    {
        if (!inheritedFields.has(fieldName))
        {
            throw new TypeError(
                `CjsSchema.hideInherited cannot hide "${fieldName}" on ${className}: ` +
                "the parent schema does not expose that field."
            );
        }
    }

    const schema = getOrCreateClassSchema(Constructor);
    for (const fieldName of fieldNames)
    {
        schema.hiddenInherited.add(fieldName);
    }
}

function getEffectiveFields(Constructor)
{
    const ordered = [];
    const byName = new Map();
    const hidden = new Set();

    for (const current of getSchemaLineage(Constructor))
    {
        const schema = CLASS_SCHEMA.get(current);
        for (const field of schema?.fields || [])
        {
            const existing = byName.get(field.name);
            if (existing)
            {
                mergeMemberMetadata(existing, field);
            }
            else
            {
                const merged = mergeMemberMetadata({ name: field.name }, field);
                ordered.push(merged);
                byName.set(field.name, merged);
            }
        }

        for (const fieldName of schema?.hiddenInherited || [])
        {
            hidden.add(fieldName);
        }
    }

    return ordered.filter(field => !hidden.has(field.name));
}

function getHiddenInheritedFieldNames(Constructor)
{
    const hidden = new Set();
    for (const current of getSchemaLineage(Constructor))
    {
        for (const fieldName of CLASS_SCHEMA.get(current)?.hiddenInherited || [])
        {
            hidden.add(fieldName);
        }
    }
    return hidden;
}

function getSchemaLineage(Constructor)
{
    const lineage = [];
    let current = Constructor;
    while (typeof current === "function")
    {
        if (CLASS_SCHEMA.has(current)) lineage.push(current);
        current = Object.getPrototypeOf(current);
    }
    return lineage.reverse();
}

function mergeMemberMetadata(target, source)
{
    for (const [namespace, value] of Object.entries(source))
    {
        if (namespace === "name") continue;
        target[namespace] = mergeNamespace(target[namespace], value);
    }
    return target;
}

function getOrCreateClassSchema(Constructor)
{
    let schema = CLASS_SCHEMA.get(Constructor);
    if (!schema)
    {
        schema = {
            className: null,
            family: null,
            sourceClass: null,
            aliases: null,
            fields: [],
            fieldsByName: new Map(),
            hiddenInherited: new Set(),
            methods: [],
            methodsByName: new Map()
        };
        CLASS_SCHEMA.set(Constructor, schema);
    }
    return schema;
}

function normalizeHiddenInheritedFields(fieldNames)
{
    if (!Array.isArray(fieldNames) || !fieldNames.length)
    {
        throw new TypeError("CjsSchema.hideInherited requires a non-empty array of field names.");
    }

    const normalized = fieldNames.map((fieldName, index) =>
    {
        if (typeof fieldName !== "string" || !fieldName.trim())
        {
            throw new TypeError(`CjsSchema.hideInherited fieldNames[${index}] must be a non-empty string.`);
        }
        return fieldName.trim();
    });

    return Object.freeze([...new Set(normalized)]);
}

function recordStage3FieldMetadata(context, namespace, value)
{
    const metadata = context?.metadata;
    if (!metadata || typeof metadata !== "object") return;

    let fields;
    if (Object.prototype.hasOwnProperty.call(metadata, STAGE3_FIELD_METADATA))
    {
        fields = metadata[STAGE3_FIELD_METADATA];
    }
    else
    {
        fields = [];
        Object.defineProperty(metadata, STAGE3_FIELD_METADATA, {
            configurable: false,
            enumerable: false,
            value: fields,
            writable: false
        });
    }

    fields.push({
        name: context.name,
        namespace,
        value
    });
}

function registerStage3FieldMetadata(Constructor, metadata)
{
    if (!metadata || typeof metadata !== "object") return;
    if (!Object.prototype.hasOwnProperty.call(metadata, STAGE3_FIELD_METADATA)) return;

    for (const field of metadata[STAGE3_FIELD_METADATA])
    {
        defineFieldMetadata(Constructor, field.name, field.namespace, field.value);
    }
}

function normalizeClassDefinition(Constructor, definition)
{
    if (typeof definition === "string")
    {
        definition = { className: definition };
    }

    const result = { ...(definition || {}) };
    if (typeof result.className !== "string" || !result.className.trim())
    {
        throw new TypeError("CjsSchema.define requires an explicit non-empty className.");
    }
    result.className = result.className.trim();
    if (!result.sourceClass && result.className) result.sourceClass = result.className;
    const aliases = [
        ...(result.aliases === undefined ? [] : Array.isArray(result.aliases) ? result.aliases : [result.aliases]),
        ...(result.alias === undefined ? [] : Array.isArray(result.alias) ? result.alias : [result.alias])
    ].filter(alias => typeof alias === "string" && alias.trim()).map(alias => alias.trim())
        .filter(alias => alias !== result.className);
    result.aliases = aliases.length ? [...new Set(aliases)] : null;
    result.fields = normalizeManualMembers(result.fields, "fields");
    result.methods = normalizeManualMembers(result.methods, "methods");
    delete result.alias;
    return result;
}

function normalizeManualMembers(members, memberType)
{
    if (members === undefined || members === null) return [];
    if (!Array.isArray(members))
    {
        throw new TypeError(`CjsSchema.define ${memberType} must be an array.`);
    }

    return members.map((member, index) =>
    {
        if (!isPlainObject(member) || typeof member.name !== "string" || !member.name.trim())
        {
            throw new TypeError(`CjsSchema.define ${memberType}[${index}] requires a non-empty name.`);
        }
        return {
            ...member,
            name: member.name.trim()
        };
    });
}

function registerClassMetadata(Constructor, schema)
{
    if (!Constructor || !schema?.className) return;

    CjsSchema.SetConstructor(schema.className, Constructor);

    for (const alias of schema.aliases || [])
    {
        CjsSchema.SetConstructor(alias, Constructor);
    }
}

function defineEnumMetadata(values, schema)
{
    if (!values || typeof values !== "object" || !schema?.name) return;

    ENUM_SCHEMA_BY_NAME.set(schema.name, schema);
    ENUM_SCHEMA_BY_OBJECT.set(values, schema);

    if (Object.isExtensible(values) && !Object.prototype.hasOwnProperty.call(values, CJS_ENUM_NAME))
    {
        Object.defineProperty(values, CJS_ENUM_NAME, {
            value: schema.name
        });
    }
}

function normalizeEnumSchema(values, definition)
{
    const type = values?.Type || values;
    const members = Array.isArray(definition.members)
        ? Object.freeze(definition.members.map(member => Object.freeze({ ...member })))
        : Object.freeze([]);
    const result = {
        name: definition.name || values?.[CJS_ENUM_NAME] || values?.Source?.name || values?.name || null,
        type,
        members
    };

    if (definition.source) result.source = definition.source;
    if (definition.family) result.family = definition.family;
    if (definition.line !== undefined && definition.line !== null) result.line = definition.line;

    return Object.freeze(result);
}

function normalizeEnumDefinition(values)
{
    if (typeof values === "string")
    {
        return Object.freeze({ enumType: values });
    }

    const type = values?.Type || (isPlainObject(values) ? values : null);
    if (type && typeof type === "object")
    {
        const result = {
            values: type
        };
        const enumType = CjsSchema.getEnumName(values);
        if (enumType) result.enumType = enumType;
        return Object.freeze(result);
    }

    return Object.freeze({ values });
}

function normalizeComponentDefinition(definition)
{
    if (!isPlainObject(definition))
    {
        throw new TypeError("CjsSchema.components requires a plain object definition.");
    }

    return Object.freeze(Object.fromEntries(Object.entries(definition).map(([swizzle, value]) => [
        normalizeSwizzle(swizzle),
        normalizeComponentEntry(value)
    ])));
}

function normalizeComponentEntry(value)
{
    if (typeof value === "string") return Object.freeze({ name: value });
    if (isPlainObject(value)) return cloneSchemaValue(value);
    return value;
}

function getComponentValue(value, swizzle)
{
    const indices = getComponentIndices(swizzle);
    if (indices.length === 1) return value?.[indices[0]];
    return indices.map(index => value?.[index]);
}

function setComponentValue(target, swizzle, value)
{
    if (!target)
    {
        throw new TypeError("CjsSchema.components.set requires a target vector.");
    }

    const indices = getComponentIndices(swizzle);
    if (indices.length === 1)
    {
        target[indices[0]] = value;
        return target;
    }

    if (!value || typeof value[Symbol.iterator] !== "function")
    {
        throw new TypeError(`CjsSchema.components.set requires an iterable value for '${swizzle}'.`);
    }

    let offset = 0;
    for (const item of value)
    {
        if (offset >= indices.length) break;
        target[indices[offset++]] = item;
    }

    if (offset !== indices.length)
    {
        throw new RangeError(`CjsSchema.components.set expected ${indices.length} values for '${swizzle}' but received ${offset}.`);
    }

    return target;
}

function getComponentIndices(swizzle)
{
    return Object.freeze([...normalizeSwizzle(swizzle)].map(componentIndex));
}

function normalizeSwizzle(swizzle)
{
    const text = String(swizzle || "").trim().toLowerCase();
    if (!text)
    {
        throw new TypeError("Component swizzle cannot be empty.");
    }

    for (const char of text)
    {
        componentIndex(char);
    }

    return text;
}

function componentIndex(char)
{
    switch (char)
    {
        case "x":
        case "r":
        case "0":
            return 0;
        case "y":
        case "g":
        case "1":
            return 1;
        case "z":
        case "b":
        case "2":
            return 2;
        case "w":
        case "a":
        case "3":
            return 3;
        default:
            throw new RangeError(`Unsupported component '${char}'.`);
    }
}

function mergeNamespace(existing, value)
{
    if (!existing) return cloneSchemaValue(value);
    if (isPlainObject(existing) && isPlainObject(value)) return Object.freeze({ ...existing, ...value });
    return cloneSchemaValue(value);
}

// Resolves @schema.enum("X") through the owning class's PascalCase static so
// exported schemas are self-describing: adds the class-scoped identity and a
// reference to the frozen member map when the static resolves.
function enrichEnumField(exported, Constructor)
{
    const enumType = exported?.enum?.enumType;
    if (!enumType) return exported;
    const members = Constructor?.[enumType];
    if (!members || typeof members !== "object") return exported;

    let owner = Constructor;
    let current = Constructor;
    while (typeof current === "function")
    {
        if (Object.prototype.hasOwnProperty.call(current, enumType))
        {
            owner = current;
            break;
        }
        current = Object.getPrototypeOf(current);
    }

    return Object.freeze({
        ...exported,
        enum: Object.freeze({
            ...exported.enum,
            identity: `${CjsSchema.getClassName(owner) || owner.name}.${enumType}`,
            members
        })
    });
}

function exportField(field, namespaces)
{
    const result = {
        name: field.name
    };

    for (const [key, value] of Object.entries(field))
    {
        if (key === "name") continue;
        if (namespaces && !namespaces.has(key)) continue;
        result[key] = cloneSchemaValue(value);
    }

    return Object.freeze(result);
}

function normalizeNamespaces(namespaces)
{
    if (!namespaces) return null;
    return new Set(Array.isArray(namespaces) ? namespaces : [namespaces]);
}

function cloneSchemaValue(value)
{
    if (Array.isArray(value)) return Object.freeze(value.map(cloneSchemaValue));
    if (isPlainObject(value)) return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneSchemaValue(item)])));
    return value;
}

function isPlainObject(value)
{
    return Boolean(value) && typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype;
}

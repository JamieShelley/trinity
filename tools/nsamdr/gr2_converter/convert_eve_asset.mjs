#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { pathToFileURL } from "node:url";
import CjsFormatGr2 from "@carbonenginejs/format-gr2";
import CjsDdsFormat from "@carbonenginejs/runtime-resource/formats/dds";
import * as BlackReader from "black-reader";
import blackClasses from "black-reader/black-classes.js";
import * as blackReaders from "black-reader/black-readers.js";

const PNG_SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const CRC = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c >>> 0;
    }
    return table;
})();

function fail(message, code = 1) {
    const error = new Error(message);
    error.exitCode = code;
    throw error;
}

function bytes(file) { return new Uint8Array(fs.readFileSync(file)); }
function parent(file) { fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true }); }
function nums(value) {
    return value && typeof value.length === "number"
        ? Array.from(value, item => Number.isFinite(+item) ? +item : 0)
        : [];
}
function crc32(data) {
    let c = 0xffffffff;
    for (const b of data) c = CRC[(c ^ b) & 255] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
    const typeBytes = Buffer.from(type), payload = Buffer.from(data);
    const out = Buffer.alloc(12 + payload.length);
    out.writeUInt32BE(payload.length, 0);
    typeBytes.copy(out, 4);
    payload.copy(out, 8);
    out.writeUInt32BE(crc32(Buffer.concat([typeBytes, payload])), 8 + payload.length);
    return out;
}
function png(width, height, rgba) {
    const raw = Buffer.alloc(height * (width * 4 + 1));
    for (let y = 0; y < height; y++) {
        const offset = y * (width * 4 + 1);
        raw[offset] = 0;
        Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, offset + 1);
    }
    const header = Buffer.alloc(13);
    header.writeUInt32BE(width, 0);
    header.writeUInt32BE(height, 4);
    header[8] = 8;
    header[9] = 6;
    return Buffer.concat([PNG_SIG, chunk("IHDR", header), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}
function to8(payload, exposure = 1) {
    const count = payload.width * payload.height * 4;
    if (payload.data instanceof Uint8Array) return new Uint8Array(payload.data.slice(0, count));
    const out = new Uint8Array(count);
    for (let i = 0; i < count; i += 4) {
        for (let c = 0; c < 3; c++) {
            const value = Math.max(Number(payload.data[i + c]) || 0, 0) * exposure;
            out[i + c] = Math.round(Math.pow(value / (1 + value), 1 / 2.2) * 255);
        }
        out[i + 3] = Math.round(Math.max(0, Math.min(1, Number(payload.data[i + 3]) || 1)) * 255);
    }
    return out;
}

function readGr2(file) {
    const input = bytes(file);
    const attempts = [
        { emit: "json", unpackTangents: true, rebuildMissingNormals: true },
        { emit: "json", unpackTangents: true },
        { emit: "json" }
    ];
    let last;
    for (const options of attempts) {
        try { return { root: CjsFormatGr2.read(input, options), options }; }
        catch (error) { last = error; }
    }
    fail(`GR2 decode failed: ${last?.message || "unknown error"}`, 20);
}

function meshScore(mesh) {
    const vertices = Math.floor(nums(mesh?.vertex?.position).length / 3);
    const triangles = (mesh?.indices || []).reduce(
        (count, group) => count + Math.floor(nums(group?.faces).length / 3), 0);
    return { vertices, triangles, score: triangles * 16 + vertices };
}

function meshFamilyKey(mesh, index) {
    const raw = String(mesh?.name || `mesh_${index}`).trim();
    // Granny files commonly bind every LOD of one render mesh to the model.
    // LOD variants are alternatives, not additive submeshes.
    const stripped = raw
        .replace(/\s+LOD(?:\s+|[_-]?)(?:\d+(?:\.\d+)?)\s*$/iu, "")
        .replace(/[_-]LOD[_-]?\d+(?:\.\d+)?\s*$/iu, "")
        .trim();
    return normalized(stripped || raw || `mesh_${index}`);
}

function isExplicitLodMesh(mesh) {
    return /(?:^|[\s_-])LOD(?:[\s_-]*\d+(?:\.\d+)?)?\s*$/iu.test(String(mesh?.name || ""));
}

function chooseRenderMesh(items) {
    return [...items].sort((left, right) => {
        // Prefer the unsuffixed production/high-detail mesh. Fall back to the
        // largest triangle/vertex payload if every candidate is LOD-labelled.
        const leftLod = isExplicitLodMesh(left.mesh) ? 1 : 0;
        const rightLod = isExplicitLodMesh(right.mesh) ? 1 : 0;
        if (leftLod !== rightLod) return leftLod - rightLod;
        if (left.quality.triangles !== right.quality.triangles) {
            return right.quality.triangles - left.quality.triangles;
        }
        if (left.quality.vertices !== right.quality.vertices) {
            return right.quality.vertices - left.quality.vertices;
        }
        return left.index - right.index;
    })[0];
}

function collapseLodAlternatives(items) {
    const families = new Map();
    for (const item of items) {
        const key = meshFamilyKey(item.mesh, item.index);
        if (!families.has(key)) families.set(key, []);
        families.get(key).push(item);
    }
    const selected = [];
    const rejected = [];
    for (const familyItems of families.values()) {
        const chosen = chooseRenderMesh(familyItems);
        selected.push(chosen);
        for (const item of familyItems) {
            if (item.index !== chosen.index) rejected.push(item);
        }
    }
    selected.sort((left, right) => left.index - right.index);
    rejected.sort((left, right) => left.index - right.index);
    return { selected, rejected };
}

function selectModelMeshes(root) {
    const meshes = Array.isArray(root?.meshes) ? root.meshes : [];
    const renderable = meshes
        .map((mesh, index) => ({ mesh, index, quality: meshScore(mesh) }))
        .filter(item => item.quality.vertices >= 3 && item.quality.triangles >= 1);
    if (!renderable.length) fail("GR2 contains no renderable triangular mesh", 21);

    let bestModel = null;
    for (const [modelIndex, model] of (root?.models || []).entries()) {
        const indices = Array.from(new Set((model?.meshBindings || [])
            .map(value => Number(value))
            .filter(value => Number.isInteger(value) && value >= 0 && value < meshes.length)));
        const bound = indices
            .map(index => renderable.find(item => item.index === index))
            .filter(Boolean);
        if (!bound.length) continue;
        const collapsed = collapseLodAlternatives(bound);
        const score = collapsed.selected.reduce((total, item) => total + item.quality.score, 0);
        if (!bestModel || score > bestModel.score) {
            bestModel = {
                modelIndex,
                modelName: String(model?.name || ""),
                selected: collapsed.selected,
                rejectedLods: collapsed.rejected,
                score
            };
        }
    }

    if (bestModel) return bestModel;
    const collapsed = collapseLodAlternatives(renderable);
    return {
        modelIndex: -1,
        modelName: "",
        selected: collapsed.selected,
        rejectedLods: collapsed.rejected,
        score: collapsed.selected.reduce((n, item) => n + item.quality.score, 0)
    };
}
function objToken(position, texcoord, normal) {
    if (texcoord !== null && normal !== null) return `${position}/${texcoord}/${normal}`;
    if (texcoord !== null) return `${position}/${texcoord}`;
    if (normal !== null) return `${position}//${normal}`;
    return `${position}`;
}
function safeName(value, fallback) {
    const result = String(value || fallback || "item").trim().replace(/[^A-Za-z0-9_.-]+/gu, "_");
    return result || fallback || "item";
}
function materialIndexFromGroup(group, fallback) {
    const name = String(group?.name || "");
    const match = /(?:^|[^a-z0-9])area[_\s-]?(\d+)/iu.exec(name);
    return match ? Number(match[1]) : fallback;
}

function gr2ToObj(input, output, summary) {
    const decoded = readGr2(input);
    const selection = selectModelMeshes(decoded.root);
    const lines = ["# NSAMDR complete model-bound GR2 to OBJ"];
    const drawRanges = [];
    let positionBase = 0, texcoordBase = 0, normalBase = 0;
    let groupIndex = 0, totalVertices = 0, totalTriangles = 0;
    let allNormals = true, allTexcoords = true;

    for (const selected of selection.selected) {
        const mesh = selected.mesh;
        const positions = nums(mesh?.vertex?.position);
        const normals = nums(mesh?.vertex?.normal);
        const texcoords = nums(mesh?.vertex?.texcoord0);
        const vertexCount = Math.floor(positions.length / 3);
        const hasNormal = normals.length >= vertexCount * 3;
        const hasTexcoord = texcoords.length >= vertexCount * 2;
        const meshName = safeName(mesh?.name, `mesh_${selected.index}`);
        const groups = Array.isArray(mesh?.indices) ? mesh.indices : [];

        lines.push(`o ${meshName}`);
        for (let i = 0; i < vertexCount; i++) {
            lines.push(`v ${positions[i * 3]} ${positions[i * 3 + 1]} ${positions[i * 3 + 2]}`);
        }
        if (hasTexcoord) {
            // The previous exporter wrote (1 - sourceV), then the preview had to
            // invert V again at runtime to match EVE. Bake that proven runtime
            // correction into the generated asset: OBJ V now equals GR2 source V.
            // The preview's flip-V control remains a debug override and defaults off.
            for (let i = 0; i < vertexCount; i++) {
                const sourceU = texcoords[i * 2];
                const sourceV = texcoords[i * 2 + 1];
                const bakedTextureV = sourceV;
                lines.push(`vt ${sourceU} ${bakedTextureV}`);
            }
        }
        if (hasNormal) {
            for (let i = 0; i < vertexCount; i++) lines.push(`vn ${normals[i * 3]} ${normals[i * 3 + 1]} ${normals[i * 3 + 2]}`);
        }

        for (let localGroupIndex = 0; localGroupIndex < groups.length; localGroupIndex++) {
            const group = groups[localGroupIndex];
            const faces = nums(group?.faces).map(Math.trunc);
            const validTriangles = [];
            for (let i = 0; i + 2 < faces.length; i += 3) {
                const a = faces[i], b = faces[i + 1], c = faces[i + 2];
                if (a < 0 || b < 0 || c < 0 || a >= vertexCount || b >= vertexCount || c >= vertexCount) continue;
                validTriangles.push([a, b, c]);
            }
            if (!validTriangles.length) continue;
            const materialIndex = materialIndexFromGroup(group, localGroupIndex);
            const name = `mesh_${selected.index}_${meshName}_area_${materialIndex}_draw_${groupIndex}`;
            const firstIndex = totalTriangles * 3;
            const triangleCount = validTriangles.length;
            lines.push(`g ${name}`);
            lines.push(`usemtl area_${materialIndex}`);
            const token = localIndex => objToken(
                positionBase + localIndex + 1,
                hasTexcoord ? texcoordBase + localIndex + 1 : null,
                hasNormal ? normalBase + localIndex + 1 : null);
            for (const [a, b, c] of validTriangles) lines.push(`f ${token(a)} ${token(b)} ${token(c)}`);
            {
                drawRanges.push({
                    groupIndex,
                    materialIndex,
                    name,
                    sourceGroupName: String(group?.name || ""),
                    meshIndex: selected.index,
                    meshName: String(mesh?.name || ""),
                    firstIndex,
                    indexCount: triangleCount * 3,
                    triangleCount
                });
                groupIndex++;
                totalTriangles += triangleCount;
            }
        }

        positionBase += vertexCount;
        if (hasTexcoord) texcoordBase += vertexCount;
        if (hasNormal) normalBase += vertexCount;
        totalVertices += vertexCount;
        allNormals = allNormals && hasNormal;
        allTexcoords = allTexcoords && hasTexcoord;
    }

    if (!totalTriangles) fail("Selected GR2 model produced no valid OBJ triangles", 22);
    parent(output);
    fs.writeFileSync(output, lines.join("\n") + "\n");
    if (summary) {
        parent(summary);
        fs.writeFileSync(summary, JSON.stringify({
            schema: "NSAMDR_GR2_CONVERSION_V5_BAKED_EVE_TEXTURE_V",
            source: path.resolve(input),
            output: path.resolve(output),
            selectedModelIndex: selection.modelIndex,
            selectedModelName: selection.modelName,
            selectedMeshIndices: selection.selected.map(item => item.index),
            rejectedLodMeshIndices: (selection.rejectedLods || []).map(item => item.index),
            selectionMode: "highest-detail-per-mesh-family",
            sourceMeshCount: decoded.root.meshes.length,
            selectedMeshCount: selection.selected.length,
            vertexCount: totalVertices,
            triangleCount: totalTriangles,
            hasNormal: allNormals,
            hasTexcoord0: allTexcoords,
            textureVConvention: "eve-gr2-source-v-baked",
            textureVTransform: "v_out = v_gr2",
            runtimeTextureVFlipRequired: false,
            parserOptions: decoded.options,
            drawRanges
        }, null, 2) + "\n");
    }
    console.log(`GR2 render model: model=${selection.modelName || selection.modelIndex} meshes=${selection.selected.length} rejectedLods=${(selection.rejectedLods || []).length} vertices=${totalVertices} triangles=${totalTriangles} draws=${drawRanges.length}`);
}

function field(value, name, fallback = null) {
    if (value instanceof Map) return value.has(name) ? value.get(name) : fallback;
    if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, name)) return value[name];
    return fallback;
}
function list(value, name) {
    const result = field(value, name, []);
    return Array.isArray(result) ? result : [];
}
function text(value, fallback = "") { return typeof value === "string" ? value.trim() : fallback; }
function normalized(value) { return text(value).toLowerCase().replace(/[^a-z0-9]+/gu, ""); }
function looseField(value, name) {
    const wanted = normalized(name);
    if (value instanceof Map) {
        for (const [key, child] of value.entries()) if (normalized(String(key)) === wanted) return child;
    } else if (value && typeof value === "object") {
        for (const [key, child] of Object.entries(value)) if (normalized(key) === wanted) return child;
    }
    return null;
}
function plain(value, seen = new WeakSet()) {
    if (value === null || value === undefined) return null;
    if (typeof value !== "object") return value;
    if (ArrayBuffer.isView(value)) return Array.from(value, Number);
    if (seen.has(value)) return null;
    seen.add(value);
    if (Array.isArray(value)) return value.map(item => plain(item, seen));
    const result = {};
    if (value instanceof Map) {
        for (const [key, child] of value.entries()) result[String(key)] = plain(child, seen);
    } else {
        for (const [key, child] of Object.entries(value)) result[key] = plain(child, seen);
    }
    return result;
}
function named(items, name) {
    const wanted = normalized(name);
    return items.find(item => normalized(field(item, "name", "")) === wanted) || null;
}
function normalizedResourcePath(value) {
    return text(value)
        .replace(/\\/gu, "/")
        .replace(/\/+/gu, "/")
        .toLowerCase()
        .replace(/^res:\//u, "")
        .replace(/^\/+/u, "");
}
function resourceBasename(value) {
    const cleaned = normalizedResourcePath(value).replace(/[?#].*$/u, "");
    return cleaned.slice(cleaned.lastIndexOf("/") + 1);
}
function resourceStem(value) {
    return resourceBasename(value).replace(/\.[^.]*$/u, "");
}
function resourceDirectoryLeaf(value) {
    const cleaned = normalizedResourcePath(value).replace(/[?#].*$/u, "");
    const directory = cleaned.slice(0, Math.max(0, cleaned.lastIndexOf("/")));
    return directory.slice(directory.lastIndexOf("/") + 1);
}
function uniqueHullMatch(matches, method, requestedHull, modelPath) {
    if (matches.length === 1) return { hull: matches[0], method };
    if (matches.length > 1) {
        const details = matches.slice(0, 12).map(item =>
            `${text(field(item, "name", "<unnamed>"))} -> ${text(field(item, "geometryResFilePath", "<no geometry>"))}`);
        fail(
            `SOF hull resolution is ambiguous (${method}) for requested=${requestedHull} model=${modelPath}: ${details.join("; ")}`,
            41,
        );
    }
    return null;
}
function resolveSofHull(hulls, requestedHull, modelPath) {
    const targetPath = normalizedResourcePath(modelPath);
    const targetBase = resourceBasename(targetPath);
    const targetStem = resourceStem(targetPath);
    const targetFamily = resourceDirectoryLeaf(targetPath);
    const requested = normalized(requestedHull);

    const geometry = hull => normalizedResourcePath(field(hull, "geometryResFilePath", ""));
    const byExactPath = targetPath
        ? hulls.filter(hull => geometry(hull) === targetPath)
        : [];
    let resolved = uniqueHullMatch(byExactPath, "exact-geometry-path", requestedHull, modelPath);
    if (resolved) return resolved;

    const byGeometryBase = targetBase
        ? hulls.filter(hull => resourceBasename(geometry(hull)) === targetBase)
        : [];
    resolved = uniqueHullMatch(byGeometryBase, "geometry-basename", requestedHull, modelPath);
    if (resolved) return resolved;

    const targetName = normalized(targetStem);
    const byTargetName = targetName
        ? hulls.filter(hull => normalized(field(hull, "name", "")) === targetName)
        : [];
    resolved = uniqueHullMatch(byTargetName, "model-stem-name", requestedHull, modelPath);
    if (resolved) return resolved;

    const byRequestedName = requested
        ? hulls.filter(hull => normalized(field(hull, "name", "")) === requested)
        : [];
    resolved = uniqueHullMatch(byRequestedName, "requested-name", requestedHull, modelPath);
    if (resolved) return resolved;

    // Some direct GR2 paths use a technical filename while SOF stores a family
    // name. Only use the containing model-family directory when it identifies a
    // single hull; never select the first partial match.
    const byFamily = targetFamily
        ? hulls.filter(hull => resourceDirectoryLeaf(geometry(hull)) === targetFamily)
        : [];
    resolved = uniqueHullMatch(byFamily, "geometry-family", requestedHull, modelPath);
    if (resolved) return resolved;

    const nearby = hulls
        .filter(hull => {
            const name = normalized(field(hull, "name", ""));
            const geometryPath = geometry(hull);
            return (requested && (name.includes(requested) || geometryPath.includes(`/${requested}/`))) ||
                (targetFamily && (name.includes(normalized(targetFamily)) || geometryPath.includes(`/${targetFamily}/`)));
        })
        .slice(0, 20)
        .map(hull => `${text(field(hull, "name", "<unnamed>"))} -> ${text(field(hull, "geometryResFilePath", "<no geometry>"))}`);
    fail(
        `SOF hull not found: requested=${requestedHull} model=${modelPath || "<missing>"}. ` +
        `Nearby candidates: ${nearby.length ? nearby.join("; ") : "none"}`,
        41,
    );
}
function parameterMap(items) {
    const out = {};
    for (const item of items || []) {
        const name = text(field(item, "name", ""));
        if (name) out[name] = plain(field(item, "value", null));
    }
    return out;
}
function textureMap(items) {
    const out = {};
    let fallbackIndex = 0;
    for (const item of items || []) {
        const resource = text(field(item, "resFilePath", ""));
        if (!resource) continue;
        const name = text(field(item, "name", ""), `Texture${fallbackIndex++}`);
        out[name] = resource.replace(/\\/gu, "/");
    }
    return out;
}
function genericStrings(items, fallbacks) {
    const values = (items || []).map(item => text(field(item, "str", item))).filter(Boolean);
    while (values.length < fallbacks.length) values.push(fallbacks[values.length]);
    return values.slice(0, fallbacks.length);
}

function ensureBlackClass(name, definitions) {
    let map = blackClasses.get(name);
    if (!map) {
        map = new Map();
        blackClasses.set(name, map);
    }
    for (const [property, reader] of Object.entries(definitions)) if (!map.has(property)) map.set(property, reader);
}
function patchCurrentSofSchema() {
    const r = blackReaders;
    ensureBlackClass("EveSOFDataFaction", {
        defaultPatternLayer2MaterialName: r.string,
        defaultPatternName: r.string
    });
    ensureBlackClass("EveSOFDataPattern", {
        applicationGroups: r.array,
        sof6: r.boolean
    });
    ensureBlackClass("EveSOFDataPatternLayerProperties", {
        projectionTypeU: r.uint,
        projectionTypeV: r.uint,
        isTargetMtl1: r.boolean,
        isTargetMtl2: r.boolean,
        isTargetMtl3: r.boolean,
        isTargetMtl4: r.boolean,
        applicableAreas: r.array
    });
    ensureBlackClass("EveSOFDataPatternApplicationGroup", {
        name: r.string,
        layer1Properties: r.object,
        layer2Properties: r.object,
        projections: r.array
    });
    ensureBlackClass("EveSOFDNADescriptor", { pattern: r.string });
    ensureBlackClass("EveSOFDataRace", {
        hullPrimaryHeatColorType: r.uint,
        hullReactorHeatColorType: r.uint
    });
    ensureBlackClass("EveSOFDataGeneric", {
        shaderPrefix: r.string,
        turretAreaType: r.uint,
        hullCategoryData: r.array
    });
    ensureBlackClass("EveSOFDataGenericHullCategory", {
        categoryName: r.string,
        reflectionMode: r.uint
    });
    ensureBlackClass("EveSOFDataHull", { modelTranslationCurvePath: r.path });
    // Current data.black adds this placement flag after extendsBoundingSphere.
    // It is a boolean in Trinity's EveSOFDataHullExtensionPlacement schema.
    ensureBlackClass("EveSOFDataHullExtensionPlacement", {
        extendsShieldEllipsoid: r.boolean
    });
}

function readBlack(file) {
    patchCurrentSofSchema();
    const input = fs.readFileSync(file);
    const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
    const context = new BlackReader.Context();
    try {
        return BlackReader.read(view, context).object;
    } catch (error) {
        const details = [error?.name, error?.message, error?.type, error?.propertyName].filter(Boolean).join(" | ");
        fail(`SOF data.black decode failed${details ? `: ${details}` : ""}`, 40);
    }
}

const AREA_TYPE_NAMES = [
    "primary", "glass", "sails", "reactor", "darkhull", "wreck",
    "rock", "monument", "ornament", "simpleprimary", "turret"
];
function areaTypeName(value) {
    if (typeof value === "number" && value >= 0 && value < AREA_TYPE_NAMES.length) return AREA_TYPE_NAMES[value];
    const result = normalized(String(value || "primary")).replace(/^type/iu, "");
    return result || "primary";
}
function areaMaterial(faction, areaType) {
    const areaTypes = field(faction, "areaTypes", null);
    const value = looseField(areaTypes, areaType);
    if (value) return value;
    return looseField(areaTypes, "primary");
}
function shaderMatches(candidate, wanted) {
    const left = normalized(candidate), right = normalized(wanted);
    return !!left && !!right && (left === right || left.endsWith(right) || right.endsWith(left));
}
function findGenericShader(generic, shader) {
    return list(generic, "areaShaders").find(item => shaderMatches(field(item, "shader", ""), shader)) || null;
}
function mergeObject(target, source, sourceName, sources) {
    for (const [key, value] of Object.entries(source || {})) {
        target[key] = value;
        if (sources) sources[key] = sourceName;
    }
}
function materialNamesForArea(faction, areaType) {
    const material = areaMaterial(faction, areaType);
    return [1, 2, 3, 4].map(index => text(field(material, `material${index}`, "")));
}
function factionAreaParameters(faction, areaName) {
    const match = named(list(faction, "areas"), areaName);
    return match ? parameterMap(list(match, "parameters")) : {};
}

function projectArea(area, passName, faction, generic, materialsByName, prefixes) {
    const areaType = areaTypeName(field(area, "areaType", 0));
    const areaName = text(field(area, "name", ""));
    const shader = text(field(area, "shader", ""));
    const genericShader = findGenericShader(generic, shader);
    const textures = {};
    mergeObject(textures, textureMap(list(genericShader, "defaultTextures")));
    mergeObject(textures, textureMap(list(area, "textures")));

    const resolvedParameters = {};
    const parameterSources = {};
    mergeObject(resolvedParameters, parameterMap(list(genericShader, "defaultParameters")), "generic-shader", parameterSources);
    mergeObject(resolvedParameters, factionAreaParameters(faction, areaName), "faction-area", parameterSources);
    mergeObject(resolvedParameters, parameterMap(list(area, "parameters")), "hull-area", parameterSources);

    const materialNames = materialNamesForArea(faction, areaType);
    const materialLibraryMatches = [];
    const unresolvedParameters = [];
    for (let index = 0; index < 4; index++) {
        const materialName = materialNames[index];
        const material = materialsByName.get(normalized(materialName));
        if (material) materialLibraryMatches.push(materialName);
        const parameters = material ? parameterMap(list(material, "parameters")) : {};
        const prefix = prefixes[index] || `Mtl${index + 1}`;
        for (const [parameterName, value] of Object.entries(parameters)) {
            const outputName = `${prefix}${parameterName}`;
            resolvedParameters[outputName] = value;
            parameterSources[outputName] = `material:${materialName}`;
        }
        for (const required of ("DiffuseColor", "FresnelColor", "Gloss")) {
            if (!Object.prototype.hasOwnProperty.call(parameters, required)) unresolvedParameters.push(`${prefix}${required}`);
        }
    }
    for (const required of ("GeneralGlowColor", "GeneralData")) {
        if (!Object.prototype.hasOwnProperty.call(resolvedParameters, required)) unresolvedParameters.push(required);
    }

    return {
        pass: passName,
        index: Number(field(area, "index", 0)) || 0,
        count: Math.max(1, Number(field(area, "count", 1)) || 1),
        areaType,
        name: areaName,
        shader,
        blockedMaterials: Number(field(area, "blockedMaterials", 0)) || 0,
        textures,
        materialNames,
        materialLibraryMatches,
        materialPrefixes: prefixes,
        resolvedParameters,
        parameterSources,
        unresolvedParameters
    };
}

function sofToJson(input, output, hullName, factionName, raceName, modelPath) {
    const root = readBlack(input);
    const hulls = list(root, "hull"), factions = list(root, "faction"), races = list(root, "race");
    const hullSelection = resolveSofHull(hulls, hullName, modelPath);
    const hull = hullSelection.hull;
    const faction = named(factions, factionName);
    const race = raceName ? named(races, raceName) : null;
    if (!faction) fail(`SOF faction not found in data.black: ${factionName}`, 42);

    const generic = field(root, "generic", null);
    const materials = list(root, "material");
    const materialsByName = new Map(materials.map(material => [normalized(field(material, "name", "")), material]));
    const prefixes = genericStrings(list(generic, "materialPrefixes"), ["Mtl1", "Mtl2", "Mtl3", "Mtl4"]);
    const areas = [];
    const passFields = [
        ["opaque", "opaqueAreas"],
        ["decal", "decalAreas"],
        ["transparent", "transparentAreas"],
        ["additive", "additiveAreas"],
        ["distortion", "distortionAreas"],
        ["depth", "depthAreas"]
    ];
    for (const [passName, property] of passFields) {
        for (const area of list(hull, property)) areas.push(projectArea(area, passName, faction, generic, materialsByName, prefixes));
    }

    const areaMaterials = {};
    for (const areaType of AREA_TYPE_NAMES) {
        const material = areaMaterial(faction, areaType);
        if (material) areaMaterials[areaType] = plain(material);
    }
    const materialLibrary = {};
    for (const material of materials) {
        const name = text(field(material, "name", ""));
        if (name) materialLibrary[name] = parameterMap(list(material, "parameters"));
    }
    const instancedMeshes = list(hull, "instancedMeshes").map(item => ({
        name: text(field(item, "name", "")),
        geometryResPath: text(field(item, "geometryResPath", "")).replace(/\\/gu, "/"),
        shader: text(field(item, "shader", "")),
        lowestLodVisible: Number(field(item, "lowestLodVisible", 0)) || 0,
        displayModifier: Number(field(item, "displayModifier", 0)) || 0,
        textures: textureMap(list(item, "textures")),
        instances: plain(field(item, "instances", []))
    }));

    const result = {
        schema: "NSAMDR_SOF_VISUALS_V2",
        source: path.resolve(input),
        requestedHull: hullName,
        requestedModelPath: modelPath,
        hullResolution: hullSelection.method,
        hull: text(field(hull, "name", hullName)),
        faction: text(field(faction, "name", factionName)),
        race: race ? text(field(race, "name", raceName)) : raceName,
        geometryResFilePath: text(field(hull, "geometryResFilePath", "")).replace(/\\/gu, "/"),
        resPathInsert: text(field(faction, "resPathInsert", "")),
        colors: plain(field(faction, "colorSet", null)) || {},
        areaMaterials,
        materialLibrary,
        areas,
        instancedMeshes,
        children: list(hull, "children").map(item => plain(item)),
        extractionDiagnostics: {
            parser: "black-reader-js",
            hullCount: hulls.length,
            factionCount: factions.length,
            materialCount: materials.length,
            selectedAreaCount: areas.length,
            selectedInstancedMeshCount: instancedMeshes.length,
            selectedModelGeometry: text(field(hull, "geometryResFilePath", ""))
        }
    };
    parent(output);
    fs.writeFileSync(output, JSON.stringify(result, null, 2) + "\n");
    console.log(`SOF hull resolution: requested=${hullName} resolved=${result.hull} method=${hullSelection.method} geometry=${result.geometryResFilePath}`);
    console.log(`SOF visual extraction: hull=${result.hull} faction=${result.faction} areas=${areas.length} instancedMeshes=${instancedMeshes.length} materials=${materials.length}`);
}

function write32(array, offset, value) {
    new DataView(array.buffer, array.byteOffset, array.byteLength).setUint32(offset, value >>> 0, true);
}
function cubeFaces(input) {
    const texture = CjsDdsFormat.read(input, { emit: "texture" });
    if (texture.dimension !== "cube" || texture.faces !== 6) return null;
    const result = [];
    for (let face = 0; face < 6; face++) {
        const subresource = texture.subresources.find(item => item.face === face && item.arrayIndex === 0 && item.mip === 0);
        if (!subresource) fail(`DDS cube missing face ${face}`, 30);
        const dataOffset = texture.metadata.dataOffset;
        const single = new Uint8Array(dataOffset + subresource.byteLength);
        single.set(input.subarray(0, dataOffset));
        single.set(input.subarray(dataOffset + subresource.offset, dataOffset + subresource.offset + subresource.byteLength), dataOffset);
        write32(single, 28, 1);
        write32(single, 112, 0);
        if (texture.metadata.hasDx10) { write32(single, 136, 0); write32(single, 140, 1); }
        const rgba = CjsDdsFormat.read(single, { emit: "rgba" });
        result.push({ width: rgba.width, height: rgba.height, data: to8(rgba) });
    }
    return result;
}
function sample(face, u, v) {
    const x = Math.max(0, Math.min(face.width - 1, (u * .5 + .5) * (face.width - 1)));
    const y = Math.max(0, Math.min(face.height - 1, (v * .5 + .5) * (face.height - 1)));
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, face.width - 1), y1 = Math.min(y0 + 1, face.height - 1);
    const tx = x - x0, ty = y - y0, out = new Uint8Array(4);
    for (let c = 0; c < 4; c++) {
        const a = face.data[(y0 * face.width + x0) * 4 + c], b = face.data[(y0 * face.width + x1) * 4 + c];
        const d = face.data[(y1 * face.width + x0) * 4 + c], e = face.data[(y1 * face.width + x1) * 4 + c];
        out[c] = Math.round((a * (1 - tx) + b * tx) * (1 - ty) + (d * (1 - tx) + e * tx) * ty);
    }
    return out;
}
function lookup(x, y, z) {
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    if (ax >= ay && ax >= az) return x >= 0 ? [0, -z / ax, -y / ax] : [1, z / ax, -y / ax];
    if (ay >= ax && ay >= az) return y >= 0 ? [2, x / ay, z / ay] : [3, x / ay, -z / ay];
    return z >= 0 ? [4, x / az, -y / az] : [5, -x / az, -y / az];
}
function equirect(faces) {
    const width = Math.max(4, Math.min(Math.max(faces[0].width, faces[0].height) * 4, 4096));
    const height = Math.floor(width / 2), out = new Uint8Array(width * height * 4);
    for (let py = 0; py < height; py++) {
        const theta = (py + .5) / height * Math.PI, st = Math.sin(theta), y = Math.cos(theta);
        for (let px = 0; px < width; px++) {
            const phi = ((px + .5) / width - .5) * Math.PI * 2, x = st * Math.sin(phi), z = st * Math.cos(phi);
            const [faceIndex, u, v] = lookup(x, y, z);
            out.set(sample(faces[faceIndex], u, v), (py * width + px) * 4);
        }
    }
    return { width, height, data: out };
}
function ddsToPng(input, output, environment) {
    const source = bytes(input), faces = environment ? cubeFaces(source) : null;
    const decoded = faces ? equirect(faces) : CjsDdsFormat.read(source, { emit: "rgba" });
    const rgba = decoded.data instanceof Uint8Array ? decoded.data : to8(decoded, environment ? 1.2 : 1);
    parent(output);
    fs.writeFileSync(output, png(decoded.width, decoded.height, rgba));
}

function usage() {
    return "Usage: convert_eve_asset.mjs <gr2-to-obj|dds-to-png|dds-to-environment-png|sof-to-json> <input> <output> [hull faction race modelPath]";
}
export function main(args = process.argv.slice(2)) {
    const [command, input, output, ...extra] = args;
    if (!command || !input || !output) fail(usage(), 2);
    if (command === "gr2-to-obj") gr2ToObj(input, output, extra[0]);
    else if (command === "dds-to-png") ddsToPng(input, output, false);
    else if (command === "dds-to-environment-png") ddsToPng(input, output, true);
    else if (command === "sof-to-json") sofToJson(input, output, extra[0] || "", extra[1] || "", extra[2] || "", extra[3] || "");
    else fail(`Unknown command ${command}\n${usage()}`, 2);
    return 0;
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
    try { process.exitCode = main(); }
    catch (error) {
        console.error(`ERROR: ${error?.message || error}`);
        process.exitCode = Number.isInteger(error?.exitCode) ? error.exitCode : 1;
    }
}

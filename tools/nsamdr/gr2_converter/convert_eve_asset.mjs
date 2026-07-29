import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { CjsGr2Format } from "@carbonenginejs/runtime-resource/formats/gr2";
import { CjsDdsFormat } from "@carbonenginejs/runtime-resource/formats/dds";
import { CjsBlackFormat } from "@carbonenginejs/runtime-resource/formats/black";

function fail(message) {
  console.error(`ERROR: ${message}`);
  process.exitCode = 1;
}

function sanitizeName(value, fallback) {
  const cleaned = String(value || fallback)
    .replace(/[^A-Za-z0-9_.-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || fallback;
}

function finite(value) {
  return Number.isFinite(value) ? value : 0;
}

function meshTriangleCount(mesh) {
  return (mesh?.indices || []).reduce((total, group) => total + Math.floor((group?.faces?.length || 0) / 3), 0);
}

function chooseHighestDetailMesh(meshes) {
  const renderable = meshes
    .map((mesh, index) => ({ mesh, index, triangles: meshTriangleCount(mesh) }))
    .filter((entry) => entry.triangles > 0 && (entry.mesh?.vertex?.position?.length || 0) >= 9);
  if (!renderable.length) throw new Error("GR2 contained no renderable meshes");

  const nonLod = renderable.filter((entry) => !/\bLOD\b/i.test(String(entry.mesh?.name || "")));
  const candidates = nonLod.length ? nonLod : renderable;
  candidates.sort((a, b) => b.triangles - a.triangles || a.index - b.index);
  return candidates[0];
}

function writeObj(inputPath, outputPath, summaryPath) {
  const bytes = fs.readFileSync(inputPath);
  const graph = CjsGr2Format.read(bytes, {
    emit: "json",
    unpackTangents: true,
    rebuildMissingNormals: true,
    rebuildMissingTangents: true,
    rebuildMissingBiNormals: true
  });

  if (!graph?.meshes?.length) throw new Error("GR2 contained no meshes");
  const selected = chooseHighestDetailMesh(graph.meshes);
  const mesh = selected.mesh;
  const meshIndex = selected.index;
  const positions = mesh?.vertex?.position || [];
  const normals = mesh?.vertex?.normal || [];
  const uvs = mesh?.vertex?.texcoord0 || [];
  const vertexCount = Math.floor(positions.length / 3);
  const normalCount = Math.floor(normals.length / 3);
  const uvCount = Math.floor(uvs.length / 2);

  if (!vertexCount) throw new Error("Selected GR2 mesh contained no vertices");
  if (uvCount !== vertexCount) {
    throw new Error(`Mesh ${meshIndex} (${mesh?.name || "unnamed"}) has ${vertexCount} vertices but ${uvCount} UV0 entries; NSAMDR needs original UVs`);
  }

  const hasNormals = normalCount === vertexCount;
  const objectName = sanitizeName(mesh?.name, `mesh_${meshIndex}`);
  const lines = [
    "# NSAMDR Granny-free EVE GR2 conversion",
    `# Source: ${path.basename(inputPath)}`,
    `# Highest-detail mesh: ${mesh?.name || objectName}`,
    "# Lower LOD meshes intentionally excluded",
    "",
    `o ${objectName}`
  ];

  for (let i = 0; i < vertexCount; i += 1) {
    const base = i * 3;
    lines.push(`v ${finite(positions[base])} ${finite(positions[base + 1])} ${finite(positions[base + 2])}`);
  }
  for (let i = 0; i < uvCount; i += 1) {
    const base = i * 2;
    lines.push(`vt ${finite(uvs[base])} ${finite(uvs[base + 1])}`);
  }
  if (hasNormals) {
    for (let i = 0; i < normalCount; i += 1) {
      const base = i * 3;
      lines.push(`vn ${finite(normals[base])} ${finite(normals[base + 1])} ${finite(normals[base + 2])}`);
    }
  }

  let totalTriangles = 0;
  let firstIndex = 0;
  const drawRanges = [];
  const groups = mesh.indices || [];
  groups.forEach((group, groupIndex) => {
    const groupName = sanitizeName(group?.name, `area_${groupIndex}`);
    const materialName = `area_${groupIndex}`;
    lines.push(`g ${objectName}_${groupName}`, `usemtl ${materialName}`);
    const faces = group?.faces || [];
    let groupTriangles = 0;
    for (let i = 0; i + 2 < faces.length; i += 3) {
      const local = [faces[i], faces[i + 1], faces[i + 2]];
      if (local.some((index) => !Number.isInteger(index) || index < 0 || index >= vertexCount)) {
        throw new Error(`Mesh ${meshIndex}, group ${groupIndex} contains an invalid triangle index`);
      }
      const tokens = local.map((index) => {
        const objIndex = index + 1;
        return hasNormals ? `${objIndex}/${objIndex}/${objIndex}` : `${objIndex}/${objIndex}`;
      });
      lines.push(`f ${tokens.join(" ")}`);
      groupTriangles += 1;
    }
    const indexCount = groupTriangles * 3;
    drawRanges.push({
      groupIndex,
      groupName: group?.name || groupName,
      materialName,
      firstIndex,
      indexCount,
      triangles: groupTriangles
    });
    firstIndex += indexCount;
    totalTriangles += groupTriangles;
  });

  if (!totalTriangles) throw new Error("GR2 conversion produced no renderable triangles");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf8");

  const summary = {
    source: path.resolve(inputPath),
    output: path.resolve(outputPath),
    grannyFileFormatRevision: graph.grannyFileFormatRevision,
    grannyFileSource: graph.grannyFileSource,
    sourceMeshCount: graph.meshes.length,
    selectedMesh: {
      index: meshIndex,
      name: mesh?.name || objectName,
      vertices: vertexCount,
      triangles: totalTriangles,
      groups: groups.length,
      hasNormals,
      hasUv0: true
    },
    excludedMeshes: graph.meshes
      .map((candidate, index) => ({ index, name: candidate?.name || `mesh_${index}`, triangles: meshTriangleCount(candidate) }))
      .filter((entry) => entry.index !== meshIndex),
    drawRanges,
    totalVertices: vertexCount,
    totalTriangles
  };
  if (summaryPath) fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  console.log(`Converted highest-detail GR2 mesh -> OBJ: ${outputPath}`);
  console.log(`Mesh=${summary.selectedMesh.name} vertices=${vertexCount} triangles=${totalTriangles} groups=${groups.length}`);
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBytes, data])), 0);
  return Buffer.concat([length, typeBytes, data, crc]);
}

function toRgba8(result) {
  if (result.data instanceof Uint8Array) return Buffer.from(result.data);
  if (result.data instanceof Float32Array) {
    const output = Buffer.alloc(result.data.length);
    for (let i = 0; i < result.data.length; i += 1) {
      const value = Math.max(0, Math.min(1, finite(result.data[i])));
      output[i] = Math.round(value * 255);
    }
    return output;
  }
  throw new Error(`Unsupported decoded DDS data type: ${result.data?.constructor?.name || typeof result.data}`);
}

function encodePng(width, height, rgba) {
  if (rgba.length !== width * height * 4) {
    throw new Error(`RGBA size mismatch: got ${rgba.length}, expected ${width * height * 4}`);
  }
  const scanlines = Buffer.alloc(height * (1 + width * 4));
  const rowBytes = width * 4;
  for (let y = 0; y < height; y += 1) {
    const dst = y * (rowBytes + 1);
    scanlines[dst] = 0;
    rgba.copy(scanlines, dst + 1, y * rowBytes, (y + 1) * rowBytes);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(scanlines, { level: 9 })),
    pngChunk("IEND", Buffer.alloc(0))
  ]);
}

function writePngFromDds(inputPath, outputPath) {
  const bytes = fs.readFileSync(inputPath);
  const decoded = CjsDdsFormat.read(bytes, { emit: "rgba", source: inputPath });
  const rgba = toRgba8(decoded);
  const png = encodePng(decoded.width, decoded.height, rgba);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, png);
  console.log(`Converted DDS -> PNG: ${outputPath}`);
  console.log(`${decoded.width}x${decoded.height} ${decoded.metadata?.pixelFormat || decoded.pixelFormat}`);
}


function makeSingleFaceDds(sourceBytes, texture, subresource) {
  const metadata = texture.metadata;
  const dataOffset = metadata.dataOffset;
  const header = Buffer.from(sourceBytes.subarray(0, dataOffset));
  header.writeUInt32LE(1, 28); // mip count
  if (header.length >= 128) {
    const caps = header.readUInt32LE(108) & ~0x00400008; // DDSCAPS_MIPMAP | DDSCAPS_COMPLEX
    header.writeUInt32LE(caps, 108);
    header.writeUInt32LE(0, 112); // caps2 cubemap/volume flags
    header.writeUInt32LE(0, 116);
    header.writeUInt32LE(0, 120);
    header.writeUInt32LE(0, 124);
  }
  if (metadata.hasDx10 && header.length >= 148) {
    header.writeUInt32LE(header.readUInt32LE(136) & ~0x4, 136); // clear TEXTURECUBE
    header.writeUInt32LE(1, 140); // one array layer
  }
  const source = Buffer.from(texture.data.buffer, texture.data.byteOffset, texture.data.byteLength);
  const faceData = source.subarray(subresource.offset, subresource.offset + subresource.byteLength);
  return Buffer.concat([header, faceData]);
}

function decodeCubeFaces(bytes, inputPath) {
  const texture = CjsDdsFormat.read(bytes, { emit: "texture", source: inputPath });
  if (!texture?.metadata?.isCube || texture.faces < 6) return null;
  const faces = [];
  for (let face = 0; face < 6; face += 1) {
    const subresource = texture.subresources.find((entry) => entry.face === face && entry.mip === 0 && entry.arrayIndex === 0);
    if (!subresource) throw new Error(`Cubemap is missing face ${face}`);
    const faceDds = makeSingleFaceDds(bytes, texture, subresource);
    const decoded = CjsDdsFormat.read(faceDds, { emit: "rgba", source: `${inputPath}#face${face}` });
    faces.push({ width: decoded.width, height: decoded.height, rgba: toRgba8(decoded) });
  }
  return faces;
}

function directionToCubeFace(x, y, z) {
  const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
  let face, u, v;
  if (ax >= ay && ax >= az) {
    if (x >= 0) { face = 0; u = -z / ax; v = -y / ax; }
    else { face = 1; u = z / ax; v = -y / ax; }
  } else if (ay >= ax && ay >= az) {
    if (y >= 0) { face = 2; u = x / ay; v = z / ay; }
    else { face = 3; u = x / ay; v = -z / ay; }
  } else if (z >= 0) {
    face = 4; u = x / az; v = -y / az;
  } else {
    face = 5; u = -x / az; v = -y / az;
  }
  return { face, u: u * 0.5 + 0.5, v: v * 0.5 + 0.5 };
}

function sampleFace(face, u, v, channel) {
  const x = Math.max(0, Math.min(face.width - 1, u * (face.width - 1)));
  const y = Math.max(0, Math.min(face.height - 1, v * (face.height - 1)));
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, face.width - 1), y1 = Math.min(y0 + 1, face.height - 1);
  const tx = x - x0, ty = y - y0;
  const at = (px, py) => face.rgba[(py * face.width + px) * 4 + channel];
  const a = at(x0, y0) * (1 - tx) + at(x1, y0) * tx;
  const b = at(x0, y1) * (1 - tx) + at(x1, y1) * tx;
  return Math.round(a * (1 - ty) + b * ty);
}

function writeEnvironmentPngFromDds(inputPath, outputPath) {
  const bytes = fs.readFileSync(inputPath);
  const faces = decodeCubeFaces(bytes, inputPath);
  if (!faces) {
    writePngFromDds(inputPath, outputPath);
    console.log("Environment source is a 2D DDS; using it directly.");
    return;
  }

  const faceWidth = faces[0].width;
  const outputWidth = Math.min(Math.max(faceWidth * 4, 1024), 4096);
  const outputHeight = Math.floor(outputWidth / 2);
  const output = Buffer.alloc(outputWidth * outputHeight * 4);
  for (let y = 0; y < outputHeight; y += 1) {
    const latitude = (0.5 - (y + 0.5) / outputHeight) * Math.PI;
    const cosLatitude = Math.cos(latitude);
    const directionY = Math.sin(latitude);
    for (let x = 0; x < outputWidth; x += 1) {
      const longitude = (((x + 0.5) / outputWidth) * 2 - 1) * Math.PI;
      const directionX = Math.sin(longitude) * cosLatitude;
      const directionZ = Math.cos(longitude) * cosLatitude;
      const mapping = directionToCubeFace(directionX, directionY, directionZ);
      const face = faces[mapping.face];
      const offset = (y * outputWidth + x) * 4;
      output[offset] = sampleFace(face, mapping.u, mapping.v, 0);
      output[offset + 1] = sampleFace(face, mapping.u, mapping.v, 1);
      output[offset + 2] = sampleFace(face, mapping.u, mapping.v, 2);
      output[offset + 3] = 255;
    }
  }
  const png = encodePng(outputWidth, outputHeight, output);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, png);
  console.log(`Converted EVE cubemap DDS -> equirectangular PNG: ${outputPath}`);
  console.log(`${outputWidth}x${outputHeight}, source face ${faces[0].width}x${faces[0].height}`);
}

const AREA_TYPE_NAMES = [
  "primary", "glass", "sails", "reactor", "darkhull", "wreck",
  "rock", "monument", "ornament", "simpleprimary", "turret"
];

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined) return [];
  if (Array.isArray(value.items)) return value.items;
  if (Array.isArray(value.values)) return value.values;
  if (typeof value === "object" && !value._type && !value.name && !value.Name) return Object.values(value);
  return [value];
}

function objectName(value) {
  return String(value?.name ?? value?._id ?? value?.Name ?? "").trim();
}

function findNamed(values, name) {
  const wanted = String(name || "").trim().toLowerCase();
  if (!wanted) return null;
  if (values && typeof values === "object" && !Array.isArray(values)) {
    for (const [key, value] of Object.entries(values)) {
      if (key.toLowerCase() === wanted) return value;
    }
  }
  return asArray(values).find((value) => objectName(value).toLowerCase() === wanted) || null;
}

function plainValue(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") return value;
  if (ArrayBuffer.isView(value)) return Array.from(value, plainValue);
  if (Array.isArray(value)) return value.map(plainValue);
  if (value instanceof Map) {
    const output = {};
    for (const [key, child] of value.entries()) output[String(key)] = plainValue(child);
    return output;
  }
  if (value instanceof Set) return Array.from(value, plainValue);
  if (typeof value === "object") {
    if (["x", "y", "z", "w"].some((key) => key in value)) {
      return ["x", "y", "z", "w"].filter((key) => key in value).map((key) => finite(Number(value[key])));
    }
    if (["r", "g", "b", "a"].some((key) => key in value)) {
      return ["r", "g", "b", "a"].filter((key) => key in value).map((key) => finite(Number(value[key])));
    }
    if ("value" in value && Object.keys(value).every((key) => ["_type", "name", "value"].includes(key))) {
      return plainValue(value.value);
    }
    const output = {};
    for (const [key, child] of Object.entries(value)) {
      if (key === "_type" || child === undefined) continue;
      output[key] = plainValue(child);
    }
    return output;
  }
  return null;
}

function parameterMap(values) {
  const output = {};
  if (values && typeof values === "object" && !Array.isArray(values) &&
      !values.name && !values.Name && !Array.isArray(values.items) && !Array.isArray(values.values)) {
    for (const [key, item] of Object.entries(values)) {
      if (key === "_type" || item === undefined) continue;
      const name = objectName(item) || key;
      output[name] = plainValue(item?.value ?? item);
    }
    return output;
  }
  for (const item of asArray(values)) {
    const name = objectName(item);
    if (name) output[name] = plainValue(item?.value ?? item);
  }
  return output;
}

function textureMap(values) {
  const output = {};
  if (values && typeof values === "object" && !Array.isArray(values) &&
      !values.name && !values.Name && !Array.isArray(values.items) && !Array.isArray(values.values)) {
    for (const [key, item] of Object.entries(values)) {
      if (key === "_type" || item === undefined || item === null) continue;
      const name = objectName(item) || key;
      const raw = typeof item === "string" ? item :
        (item?.resFilePath ?? item?.resourcePath ?? item?.path ?? item?.value ?? "");
      const resource = String(raw).trim();
      if (name && resource) output[name] = resource.replace(/\\/g, "/");
    }
    return output;
  }
  for (const item of asArray(values)) {
    const name = objectName(item);
    const resource = String(item?.resFilePath ?? item?.resourcePath ?? item?.path ?? item?.value ?? "").trim();
    if (name && resource) output[name] = resource.replace(/\\/g, "/");
  }
  return output;
}

function normalizeAreaType(value) {
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase().replace(/^type_/, "");
    if (/^\d+$/.test(normalized)) {
      const numeric = Number(normalized);
      return AREA_TYPE_NAMES[numeric] || `type_${numeric}`;
    }
    return normalized;
  }
  const numeric = Number(value);
  return AREA_TYPE_NAMES[numeric] || `type_${Number.isFinite(numeric) ? numeric : "unknown"}`;
}

function findGenericShader(generic, shaderPath, passName) {
  const lists = passName === "decal"
    ? [generic?.decalShaders, generic?.areaShaders]
    : [generic?.areaShaders, generic?.decalShaders];
  const wanted = String(shaderPath || "").toLowerCase();
  const wantedBase = path.basename(wanted);
  for (const list of lists) {
    const shaders = asArray(list);
    const exact = shaders.find((item) => String(item?.shader || "").toLowerCase() === wanted);
    if (exact) return exact;
    const basename = shaders.find((item) => path.basename(String(item?.shader || "").toLowerCase()) === wantedBase);
    if (basename) return basename;
  }
  return null;
}

function stringList(value) {
  return asArray(value)
    .map((item) => typeof item === "string" ? item : String(item?.str ?? item?.value ?? item?.name ?? ""))
    .map((item) => item.trim())
    .filter(Boolean);
}

function normalKey(value) {
  return String(value ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function findStringifiedObject(value, pathName = "$", seen = new WeakSet()) {
  if (value === "[object Object]") return pathName;
  if (value === null || typeof value !== "object") return "";
  if (seen.has(value)) return "";
  seen.add(value);
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      const found = findStringifiedObject(value[index], `${pathName}[${index}]`, seen);
      if (found) return found;
    }
  } else {
    for (const [key, child] of Object.entries(value)) {
      const found = findStringifiedObject(child, `${pathName}.${key}`, seen);
      if (found) return found;
    }
  }
  return "";
}

function mapLookup(object, name) {
  if (!object || typeof object !== "object") return undefined;
  const wanted = normalKey(name);
  for (const [key, value] of Object.entries(object)) {
    if (normalKey(key) === wanted) return value;
  }
  return undefined;
}

function areaMaterialMap(areaTypes) {
  const source = areaTypes?.materials && typeof areaTypes.materials === "object"
    ? areaTypes.materials
    : areaTypes;
  const output = {};
  if (source && typeof source === "object" && !Array.isArray(source)) {
    for (const [key, value] of Object.entries(source)) {
      if (key === "_type" || value === null || value === undefined) continue;
      const normalized = plainValue(value);
      if (normalized && typeof normalized === "object" && mapLookup(normalized, "colorType") === undefined) {
        // EveSOFDataAreaMaterial.colorType defaults to primary (0). Document
        // mode preserves only persisted fields, so restore the class default.
        normalized.colorType = 0;
      }
      output[normalizeAreaType(key)] = normalized;
    }
  } else {
    asArray(source).forEach((value, index) => {
      if (!value) return;
      const key = normalizeAreaType(value?.areaType ?? value?.type ?? index);
      const normalized = plainValue(value);
      if (normalized && typeof normalized === "object" && mapLookup(normalized, "colorType") === undefined) {
        normalized.colorType = 0;
      }
      output[key] = normalized;
    });
  }
  return output;
}

function mergeAreaMaterial(base, override) {
  const baseValue = base && typeof base === "object" ? plainValue(base) : {};
  const overrideValue = override && typeof override === "object" ? plainValue(override) : {};
  return { ...baseValue, ...overrideValue };
}

function areaMaterialFor(materials, areaType) {
  const primary = mapLookup(materials, "primary") || null;
  const specific = mapLookup(materials, areaType) || null;
  if (areaType === "primary") return primary;
  if (primary && specific) return mergeAreaMaterial(primary, specific);
  return specific || primary || null;
}

function materialNameValue(value, preferredKey = "") {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (!value || typeof value !== "object") return "";

  // Black document output preserves indexed source fields. A field such as
  // material1 can therefore arrive as { material1: "black_satin_enamel" }.
  // Unwrap those source-shape containers without flattening unrelated objects.
  for (const key of [preferredKey, "value", "name", "str"]) {
    if (!key) continue;
    const nested = mapLookup(value, key);
    if (typeof nested === "string" && nested.trim()) return nested.trim();
  }
  const entries = Object.entries(value).filter(([key, child]) => key !== "_type" && child !== undefined && child !== null);
  if (entries.length === 1) {
    return materialNameValue(entries[0][1], entries[0][0]);
  }
  return "";
}

function materialNameFromArea(areaMaterial, slot) {
  if (!areaMaterial || typeof areaMaterial !== "object") return "";

  const directCandidates = [
    `material${slot + 1}`,
    `mtl${slot + 1}`,
    `m_material${slot + 1}`,
    String(slot),
  ];
  for (const candidate of directCandidates) {
    const value = mapLookup(areaMaterial, candidate);
    const name = materialNameValue(value, candidate);
    if (name) return name;
  }

  for (const collectionName of ["material", "materials", "m_material"]) {
    const collection = mapLookup(areaMaterial, collectionName);
    if (Array.isArray(collection)) {
      const name = materialNameValue(collection[slot], `material${slot + 1}`);
      if (name) return name;
    } else if (collection && typeof collection === "object") {
      for (const candidate of [String(slot), `material${slot + 1}`, `mtl${slot + 1}`]) {
        const name = materialNameValue(mapLookup(collection, candidate), candidate);
        if (name) return name;
      }
    }
  }
  return "";
}

function materialParameter(materialLibrary, materialName, parameterName) {
  if (!materialName) return null;
  const material = mapLookup(materialLibrary, materialName) || findNamed(materialLibrary, materialName);
  if (!material) return null;
  const parameters = parameterMap(material?.parameters ?? material?.parameter ?? material);
  return plainValue(mapLookup(parameters, parameterName));
}

function buildMaterialLibrary(values) {
  const output = {};
  const addMaterial = (material, fallbackName = "") => {
    if (material === null || material === undefined) return;
    const name = objectName(material) || String(fallbackName || "").trim();
    if (!name) return;
    output[name] = {
      name,
      parameters: parameterMap(material?.parameters ?? material?.parameter ?? material),
    };
  };

  if (values && typeof values === "object" && !Array.isArray(values) &&
      !values.name && !values.Name && !Array.isArray(values.items) && !Array.isArray(values.values)) {
    for (const [key, material] of Object.entries(values)) {
      if (key === "_type") continue;
      addMaterial(material, key);
    }
  } else {
    for (const material of asArray(values)) addMaterial(material);
  }
  return output;
}

const COLOR_TYPE_NAMES = [
  "primary", "secondary", "tertiary", "black", "white", "yellow", "orange", "red", "blue", "green", "cyan", "fire",
  "hull", "glass", "reactor", "darkhull", "booster", "killmark", "primarylight", "secondarylight", "tertiarylight", "whitelight",
  "primaryhologram", "secondaryhologram", "tertiaryhologram", "state0", "state1", "state2", "state3", "statevulnerable",
  "stateinvulnerable", "primaryforcefield", "secondaryforcefield", "primarybanner", "primaryfx", "secondaryfx", "primaryspotlight",
  "secondaryspotlight", "tertiaryspotlight", "primarybillboard", "primarywarpfx", "primaryattackfx", "primarysiegefx", "primarydockedfx"
];

function colorSetValue(colorSet, colorType) {
  if (colorType === null || colorType === undefined) return null;
  const source = colorSet?.colors && typeof colorSet.colors === "object"
    ? colorSet.colors
    : colorSet;
  const numeric = Number(colorType);
  const candidates = [];
  if (Number.isInteger(numeric) && numeric >= 0 && numeric < COLOR_TYPE_NAMES.length) {
    candidates.push(COLOR_TYPE_NAMES[numeric], `type_${COLOR_TYPE_NAMES[numeric]}`, String(numeric));
    if (Array.isArray(source) && source[numeric] !== undefined) return plainValue(source[numeric]);
  }
  if (typeof colorType === "string") {
    candidates.push(colorType, colorType.replace(/^TYPE_/i, ""));
  }
  for (const candidate of candidates) {
    const found = mapLookup(source, candidate);
    if (found !== undefined) return plainValue(found);
  }
  return null;
}

function scaleVector4(value, multiplier) {
  const vector = plainValue(value);
  if (!Array.isArray(vector)) return null;
  return [0, 1, 2, 3].map((index) => finite(Number(vector[index] ?? (index === 3 ? 1 : 0))) * Number(multiplier[index] ?? 1));
}

function materialUsageSlots(faction) {
  return [0, 1, 2, 3].map((fallback, slot) => {
    const value = Number(faction?.[`materialUsageMtl${slot + 1}`]);
    return Number.isInteger(value) && value >= 0 && value < 4 ? value : fallback;
  });
}

function resolveAreaVisuals(area, passName, generic, faction, race, materialLibrary, factionAreaMaterials, raceAreaMaterials) {
  const shader = String(area?.shader || "").replace(/\\/g, "/");
  const shaderDefinition = findGenericShader(generic, shader, passName);
  const areaType = normalizeAreaType(area?.areaType ?? 0);
  const blockedMaterials = Number(area?.blockedMaterials || 0);
  const materialPrefixes = stringList(generic?.materialPrefixes);
  const fallbackPrefixes = ["Mtl1", "Mtl2", "Mtl3", "Mtl4"];
  const prefixes = materialPrefixes.length >= 4 ? materialPrefixes : fallbackPrefixes;
  const areaParameters = parameterMap(area?.parameters);
  const genericArea = mapLookup(generic?.hullAreas, objectName(area)) ?? mapLookup(generic?.hullAreas, areaType);
  const genericAreaParameters = parameterMap(genericArea?.parameters ?? genericArea);
  const defaultParameters = parameterMap(shaderDefinition?.defaultParameters);
  const parameterNames = new Set(stringList(shaderDefinition?.parameters));
  for (const name of Object.keys(areaParameters)) parameterNames.add(name);
  for (const name of Object.keys(defaultParameters)) parameterNames.add(name);

  // These are the visible quad material inputs used by the EVE hull shader. Add
  // them even when an older data.black omits the parameter list from generic data.
  for (let slot = 0; slot < 4; slot += 1) {
    parameterNames.add(`${prefixes[slot]}DiffuseColor`);
    parameterNames.add(`${prefixes[slot]}FresnelColor`);
    parameterNames.add(`${prefixes[slot]}Gloss`);
  }
  parameterNames.add("GeneralGlowColor");

  const raceArea = (areaType === "primary" || areaType === "reactor")
    ? areaMaterialFor(raceAreaMaterials, areaType)
    : null;
  const factionArea = areaMaterialFor(factionAreaMaterials, areaType);
  const raceAreaParameters = parameterMap(raceArea?.parameters);
  const factionAreaParameters = parameterMap(factionArea?.parameters);
  const usageSlots = materialUsageSlots(faction);
  const materialNames = [];
  for (let slot = 0; slot < 4; slot += 1) {
    const sourceSlot = usageSlots[slot];
    materialNames.push(materialNameFromArea(raceArea, sourceSlot) || materialNameFromArea(factionArea, sourceSlot));
  }

  const resolvedParameters = {};
  const parameterSources = {};
  const unresolvedParameters = [];
  for (const parameterName of parameterNames) {
    let value;
    let source = "";
    let materialSlot = -1;
    let shortName = parameterName;
    for (let slot = 0; slot < prefixes.length; slot += 1) {
      if (parameterName.toLowerCase().startsWith(prefixes[slot].toLowerCase())) {
        materialSlot = slot;
        shortName = parameterName.slice(prefixes[slot].length);
        break;
      }
    }

    if (materialSlot >= 0 && (blockedMaterials & (1 << materialSlot)) === 0) {
      const sourceSlot = usageSlots[materialSlot];
      const raceMaterialName = materialNameFromArea(raceArea, sourceSlot);
      const factionMaterialName = materialNameFromArea(factionArea, sourceSlot);
      for (const [materialName, label] of [[raceMaterialName, "race material"], [factionMaterialName, "faction material"]]) {
        if (!materialName) continue;
        const candidate = materialParameter(materialLibrary, materialName, shortName);
        if (candidate !== null && candidate !== undefined) {
          value = candidate;
          source = `${label}:${materialName}`;
          break;
        }
      }
    } else if (materialSlot < 0 && normalKey(parameterName) === "generalglowcolor") {
      const factionColorType = mapLookup(factionArea, "colorType");
      const glowColor = colorSetValue(faction?.colorSet, factionColorType);
      if (glowColor !== null && glowColor !== undefined) {
        // Match the SOF hull path: faction area colour multiplied by the
        // standard GeneralGlowColor intensity multiplier.
        value = scaleVector4(glowColor, [10, 10, 10, 1]);
        source = "faction colorSet";
      }
    } else if (materialSlot < 0 && normalKey(parameterName) === "generalheatglowcolor") {
      const heatColor = race?.booster?.glowColor;
      if (heatColor !== null && heatColor !== undefined) {
        value = scaleVector4(heatColor, [100, 100, 100, 1]);
        source = "race booster glowColor";
      }
    }

    for (const [parameters, label] of [
      [genericAreaParameters, "generic hull area"],
      [raceAreaParameters, "race area"],
      [factionAreaParameters, "faction area"],
      [areaParameters, "hull area"],
      [defaultParameters, "generic shader default"],
    ]) {
      if (value !== null && value !== undefined) break;
      const candidate = mapLookup(parameters, parameterName);
      if (candidate !== undefined) {
        value = plainValue(candidate);
        source = label;
      }
    }

    if (value !== null && value !== undefined) {
      resolvedParameters[parameterName] = value;
      parameterSources[parameterName] = source;
    } else {
      unresolvedParameters.push(parameterName);
    }
  }

  return {
    shader,
    shaderDefinition: String(shaderDefinition?.shader || ""),
    areaType,
    blockedMaterials,
    materialPrefixes: prefixes.slice(0, 4),
    materialNames,
    resolvedParameters,
    parameterSources,
    unresolvedParameters,
    materialLibraryMatches: materialNames.map((name) => Boolean(name && (mapLookup(materialLibrary, name) || findNamed(materialLibrary, name)))),
    defaultTextures: textureMap(shaderDefinition?.defaultTextures)
  };
}

function areaManifest(area, passName, generic, faction, race, materialLibrary, factionAreaMaterials, raceAreaMaterials) {
  const visuals = resolveAreaVisuals(area, passName, generic, faction, race, materialLibrary, factionAreaMaterials, raceAreaMaterials);
  return {
    pass: passName,
    index: Number(area?.index || 0),
    count: Math.max(1, Number(area?.count || 1)),
    name: objectName(area),
    areaType: visuals.areaType,
    shader: visuals.shader,
    shaderDefinition: visuals.shaderDefinition,
    blockedMaterials: visuals.blockedMaterials,
    textures: { ...visuals.defaultTextures, ...textureMap(area?.textures) },
    parameters: parameterMap(area?.parameters),
    resolvedParameters: visuals.resolvedParameters,
    parameterSources: visuals.parameterSources,
    unresolvedParameters: visuals.unresolvedParameters,
    materialPrefixes: visuals.materialPrefixes,
    materialNames: visuals.materialNames,
    materialLibraryMatches: visuals.materialLibraryMatches
  };
}

function isBlackDocumentRef(value) {
  return Boolean(value && typeof value === "object" && Number.isInteger(value.$ref));
}

function createBlackDocumentResolver(document) {
  const nodes = Array.isArray(document?.nodes) ? document.nodes : [];
  const byId = new Map(nodes.map((node) => [Number(node.id), node]));
  const byKind = new Map();
  for (const node of nodes) {
    const list = byKind.get(node.kind) || [];
    list.push(node);
    byKind.set(node.kind, list);
  }

  function resolveValue(value, cache = new Map()) {
    if (isBlackDocumentRef(value)) return resolveNode(value.$ref, cache);
    if (ArrayBuffer.isView(value)) return Array.from(value);
    if (Array.isArray(value)) return value.map((child) => resolveValue(child, cache));
    if (!value || typeof value !== "object") return value;
    const output = {};
    for (const [key, child] of Object.entries(value)) output[key] = resolveValue(child, cache);
    return output;
  }

  function resolveNode(id, cache = new Map()) {
    const numericId = Number(id);
    if (cache.has(numericId)) return cache.get(numericId);
    const node = byId.get(numericId);
    if (!node) throw new Error(`Black document reference ${numericId} was not found`);
    const output = { _sourceClassName: node.kind };
    cache.set(numericId, output);
    for (const [key, child] of Object.entries(node.fields || {})) {
      output[key] = resolveValue(child, cache);
    }
    return output;
  }

  function nodeByName(kind, name) {
    const wanted = String(name || "").trim().toLowerCase();
    if (!wanted) return null;
    return (byKind.get(kind) || []).find((node) => String(node?.fields?.name ?? node?.fields?.Name ?? "").trim().toLowerCase() === wanted) || null;
  }

  return { byKind, resolveValue, resolveNode, nodeByName };
}

function resolveDocumentHull(resolver, name) {
  const node = resolver.nodeByName("EveSOFDataHull", name);
  if (!node) return null;
  const fields = node.fields || {};
  const hull = {
    _sourceClassName: node.kind,
    name: fields.name,
    geometryResFilePath: fields.geometryResFilePath,
  };
  for (const field of ["opaqueAreas", "decalAreas", "transparentAreas", "additiveAreas"]) {
    hull[field] = resolver.resolveValue(fields[field] || []);
  }
  return hull;
}

function resolveDocumentFaction(resolver, name) {
  const node = resolver.nodeByName("EveSOFDataFaction", name);
  if (!node) return null;
  const fields = node.fields || {};
  const faction = {
    _sourceClassName: node.kind,
    name: fields.name,
    resPathInsert: fields.resPathInsert,
    areaTypes: resolver.resolveValue(fields.areaTypes),
    colorSet: resolver.resolveValue(fields.colorSet),
  };
  for (let slot = 1; slot <= 4; slot += 1) {
    const key = `materialUsageMtl${slot}`;
    if (fields[key] !== undefined) faction[key] = fields[key];
  }
  return faction;
}

function resolveDocumentRace(resolver, name) {
  const node = resolver.nodeByName("EveSOFDataRace", name);
  if (!node) return null;
  const fields = node.fields || {};
  return {
    _sourceClassName: node.kind,
    name: fields.name,
    booster: resolver.resolveValue(fields.booster),
    areaTypes: resolver.resolveValue(fields.areaTypes),
  };
}

function resolveDocumentGeneric(resolver) {
  const node = (resolver.byKind.get("EveSOFDataGeneric") || [])[0] || null;
  if (!node) return {};
  const fields = node.fields || {};
  return {
    _sourceClassName: node.kind,
    materialPrefixes: resolver.resolveValue(fields.materialPrefixes || []),
    areaShaders: resolver.resolveValue(fields.areaShaders || []),
    decalShaders: resolver.resolveValue(fields.decalShaders || []),
    hullAreas: resolver.resolveValue(fields.hullAreas || {}),
  };
}

function resolveDocumentMaterialLibrary(resolver) {
  const materials = [];
  for (const node of resolver.byKind.get("EveSOFDataMaterial") || []) {
    materials.push({
      _sourceClassName: node.kind,
      name: node.fields?.name || "",
      parameters: resolver.resolveValue(node.fields?.parameters || []),
    });
  }
  return materials;
}

function raceNameFromGeometryPath(geometryPath) {
  const normalized = String(geometryPath || "").replace(/\\/g, "/").toLowerCase();
  const match = normalized.match(/(?:^|\/)model\/ship\/([^/]+)\//);
  return match ? match[1] : "";
}

function extractSof(inputPath, outputPath, hullName, factionName, raceName) {
  const bytes = fs.readFileSync(inputPath);

  // Runtime hydration normalizes fields through the canonical schema. Some
  // current EVE source shapes intentionally reuse indexed field names (for
  // example EveSOFDataAreaMaterial.material1), and runtime normalization can
  // coerce those source containers to "[object Object]". Document mode keeps
  // the lossless source graph and numeric references. Resolve only the selected
  // hull/faction/race plus generic shaders and materials into plain objects.
  const document = CjsBlackFormat.readDocument(bytes);
  const resolver = createBlackDocumentResolver(document);
  const hull = resolveDocumentHull(resolver, hullName);
  const faction = resolveDocumentFaction(resolver, factionName);
  if (!hull) throw new Error(`SOF hull not found: ${hullName}`);
  if (!faction) throw new Error(`SOF faction not found: ${factionName}`);

  const requestedRace = String(raceName || "").trim() || raceNameFromGeometryPath(hull?.geometryResFilePath);
  let race = resolveDocumentRace(resolver, requestedRace);
  let raceSource = requestedRace;
  if (!race) {
    const factionNameLower = objectName(faction).toLowerCase();
    for (const node of resolver.byKind.get("EveSOFDataRace") || []) {
      const candidate = String(node?.fields?.name || "").trim().toLowerCase();
      if (candidate && factionNameLower.startsWith(candidate)) {
        race = resolveDocumentRace(resolver, candidate);
        raceSource = candidate;
        break;
      }
    }
  }
  if (!race) {
    throw new Error(`SOF race could not be derived for hull ${hullName} (${hull?.geometryResFilePath || "no geometry path"})`);
  }

  const generic = resolveDocumentGeneric(resolver);
  const materialLibrary = buildMaterialLibrary(resolveDocumentMaterialLibrary(resolver));

  const factionAreaMaterials = areaMaterialMap(faction?.areaTypes);
  const raceAreaMaterials = areaMaterialMap(race?.areaTypes);
  const areas = [];
  for (const [field, passName] of [
    ["opaqueAreas", "opaque"],
    ["decalAreas", "decal"],
    ["transparentAreas", "transparent"],
    ["additiveAreas", "additive"]
  ]) {
    for (const area of asArray(hull?.[field])) {
      areas.push(areaManifest(area, passName, generic, faction, race, materialLibrary, factionAreaMaterials, raceAreaMaterials));
    }
  }

  const required = areas.reduce((total, area) => total + 12 + (area.textures?.AlbedoMap ? 1 : 0) + (area.textures?.NormalMap ? 1 : 0), 0);
  const resolved = areas.reduce((total, area) => {
    let count = 0;
    for (let slot = 1; slot <= 4; slot += 1) {
      if (mapLookup(area.resolvedParameters, `Mtl${slot}DiffuseColor`) !== undefined) count += 1;
      if (mapLookup(area.resolvedParameters, `Mtl${slot}FresnelColor`) !== undefined) count += 1;
      if (mapLookup(area.resolvedParameters, `Mtl${slot}Gloss`) !== undefined) count += 1;
    }
    if (area.textures?.AlbedoMap) count += 1;
    if (area.textures?.NormalMap) count += 1;
    return total + count;
  }, 0);

  const result = {
    source: path.resolve(inputPath),
    hull: objectName(hull),
    faction: objectName(faction),
    race: objectName(race) || raceSource,
    raceSource: String(raceName || "").trim() ? "selectedShip" : "hullGeometry",
    geometry: String(hull?.geometryResFilePath || "").replace(/\\/g, "/"),
    resPathInsert: String(faction?.resPathInsert || ""),
    colors: plainValue(faction?.colorSet || {}),
    materialPrefixes: stringList(generic?.materialPrefixes),
    areaMaterials: factionAreaMaterials,
    raceAreaMaterials,
    materialLibrary,
    extractionDiagnostics: {
      blackGraphMode: "document-resolved",
      factionAreaTypes: Object.keys(factionAreaMaterials),
      factionPrimaryMaterialNames: [0, 1, 2, 3].map((slot) => materialNameFromArea(areaMaterialFor(factionAreaMaterials, "primary"), slot)),
      factionMaterialUsage: [
        Number(faction?.materialUsageMtl1 ?? 0),
        Number(faction?.materialUsageMtl2 ?? 1),
        Number(faction?.materialUsageMtl3 ?? 2),
        Number(faction?.materialUsageMtl4 ?? 3)
      ],
      blackReports: plainValue(document?.reports || [])
    },
    areas,
    baselineCompleteness: {
      resolved,
      required,
      ratio: required > 0 ? resolved / required : 0,
      neuralAllowed: required > 0 && resolved === required
    }
  };
  const corruptPath = findStringifiedObject(result);
  if (corruptPath) {
    throw new Error(`Black visual graph contains a coerced [object Object] value at ${corruptPath}`);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(`Extracted SOF visual manifest: ${outputPath}`);
  console.log(`Hull=${result.hull} faction=${result.faction} areas=${areas.length} baseline=${resolved}/${required}`);
}

function runSelfTest() {
  const testDocument = {
    nodes: [
      { id: 1, kind: "Root", fields: { child: { $ref: 2 } } },
      { id: 2, kind: "EveSOFDataAreaMaterial", fields: { material1: { material1: "test_mtl_1" } } },
    ]
  };
  const testResolver = createBlackDocumentResolver(testDocument);
  const hydrated = testResolver.resolveNode(1);
  if (materialNameFromArea(hydrated.child, 0) !== "test_mtl_1") {
    throw new Error("Black document reference/material unwrapping self-test failed");
  }
  if (findStringifiedObject({ nested: { bad: "[object Object]" } }) !== "$.nested.bad") {
    throw new Error("stringified-object detection self-test failed");
  }

  const parameters = parameterMap({
    Mtl1DiffuseColor: [0.1, 0.2, 0.3, 1.0],
    Mtl1Gloss: { value: 0.7 },
  });
  if (!Array.isArray(parameters.Mtl1DiffuseColor) || parameters.Mtl1Gloss !== 0.7) {
    throw new Error("parameterMap object-map self-test failed");
  }
  const textures = textureMap({
    AlbedoMap: "res:/ship/test_ar.dds",
    NormalMap: { resFilePath: "res:\\ship\\test_no.dds" },
    PmdgMap: { value: "res:/ship/test_pmdg.dds" },
  });
  if (textures.AlbedoMap !== "res:/ship/test_ar.dds" ||
      textures.NormalMap !== "res:/ship/test_no.dds" ||
      textures.PmdgMap !== "res:/ship/test_pmdg.dds") {
    throw new Error(`textureMap object-map self-test failed: ${JSON.stringify(textures)}`);
  }

  const factionAreas = areaMaterialMap({ materials: {
    0: { material1: "test_mtl_1", material2: "test_mtl_2", material3: "test_mtl_3", material4: "test_mtl_4" }
  }});
  const primary = areaMaterialFor(factionAreas, "primary");
  if (!primary || materialNameFromArea(primary, 0) !== "test_mtl_1" || normalizeAreaType("0") !== "primary") {
    throw new Error(`SOF area material self-test failed: ${JSON.stringify(factionAreas)}`);
  }
  const inherited = areaMaterialFor({
    primary: { material1: "base_1", material2: "base_2", colorType: 1 },
    reactor: { material2: "reactor_2", colorType: 14 }
  }, "reactor");
  if (materialNameFromArea(inherited, 0) !== "base_1" || materialNameFromArea(inherited, 1) !== "reactor_2" || mapLookup(inherited, "colorType") !== 14) {
    throw new Error(`SOF primary-area inheritance self-test failed: ${JSON.stringify(inherited)}`);
  }
  const nestedColor = colorSetValue({ colors: { primary: [0.1, 0.2, 0.3, 1] } }, 0);
  if (!Array.isArray(nestedColor) || nestedColor[0] !== 0.1) {
    throw new Error(`SOF nested faction colour self-test failed: ${JSON.stringify(nestedColor)}`);
  }

  const library = buildMaterialLibrary([
    { name: "test_mtl_1", parameters: [
      { name: "DiffuseColor", value: [0.2, 0.3, 0.4, 1.0] },
      { name: "FresnelColor", value: [0.05, 0.05, 0.05, 1.0] },
      { name: "Gloss", value: [0.65, 0.0, 0.0, 0.0] },
    ] }
  ]);
  const diffuse = materialParameter(library, "test_mtl_1", "DiffuseColor");
  const gloss = materialParameter(library, "test_mtl_1", "Gloss");
  if (!Array.isArray(diffuse) || diffuse[0] !== 0.2 || !Array.isArray(gloss) || gloss[0] !== 0.65) {
    throw new Error(`SOF material library self-test failed: ${JSON.stringify(library)}`);
  }
  console.log("SOF map and material resolution self-test passed");
}

function usage() {
  console.log("Usage:");
  console.log("  node convert_eve_asset.mjs gr2-to-obj <input.gr2> <output.obj> [summary.json]");
  console.log("  node convert_eve_asset.mjs dds-to-png <input.dds> <output.png>");
  console.log("  node convert_eve_asset.mjs dds-to-environment-png <input.dds> <output.png>");
  console.log("  node convert_eve_asset.mjs sof-to-json <data.black> <output.json> <hull> <faction> [race]");
  console.log("  node convert_eve_asset.mjs self-test _ _");
}

try {
  const [command, inputPath, outputPath, arg3, arg4, arg5] = process.argv.slice(2);
  if (command === "self-test") {
    runSelfTest();
  } else if (!command || !inputPath || !outputPath) {
    usage();
    process.exitCode = 2;
  } else if (command === "gr2-to-obj") {
    writeObj(inputPath, outputPath, arg3);
  } else if (command === "dds-to-png") {
    writePngFromDds(inputPath, outputPath);
  } else if (command === "dds-to-environment-png") {
    writeEnvironmentPngFromDds(inputPath, outputPath);
  } else if (command === "sof-to-json") {
    if (!arg3 || !arg4) throw new Error("sof-to-json requires hull and faction names");
    extractSof(inputPath, outputPath, arg3, arg4, arg5 || "");
  } else {
    usage();
    fail(`Unknown command ${command}`);
  }
} catch (error) {
  fail(error?.stack || String(error));
}

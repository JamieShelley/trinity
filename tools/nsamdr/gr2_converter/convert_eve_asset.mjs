import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { CjsGr2Format } from "@carbonenginejs/runtime-resource/formats/gr2";
import { CjsDdsFormat } from "@carbonenginejs/runtime-resource/formats/dds";

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

function writeObj(inputPath, outputPath, summaryPath) {
  const bytes = fs.readFileSync(inputPath);
  const graph = CjsGr2Format.read(bytes, {
    emit: "json",
    unpackTangents: true,
    rebuildMissingNormals: true,
    rebuildMissingTangents: true,
    rebuildMissingBiNormals: true
  });

  if (!graph?.meshes?.length) {
    throw new Error("GR2 contained no meshes");
  }

  const lines = [
    "# NSAMDR Granny-free EVE GR2 conversion",
    `# Source: ${path.basename(inputPath)}`,
    `# Meshes: ${graph.meshes.length}`
  ];

  let vertexOffset = 0;
  let uvOffset = 0;
  let normalOffset = 0;
  let totalVertices = 0;
  let totalTriangles = 0;
  const meshSummaries = [];

  graph.meshes.forEach((mesh, meshIndex) => {
    const positions = mesh?.vertex?.position || [];
    const normals = mesh?.vertex?.normal || [];
    const uvs = mesh?.vertex?.texcoord0 || [];
    const vertexCount = Math.floor(positions.length / 3);
    const normalCount = Math.floor(normals.length / 3);
    const uvCount = Math.floor(uvs.length / 2);

    if (!vertexCount) return;
    if (uvCount !== vertexCount) {
      throw new Error(
        `Mesh ${meshIndex} (${mesh?.name || "unnamed"}) has ${vertexCount} vertices but ${uvCount} UV0 entries; NSAMDR needs original UVs`
      );
    }

    const hasNormals = normalCount === vertexCount;
    const objectName = sanitizeName(mesh?.name, `mesh_${meshIndex}`);
    lines.push("", `o ${objectName}`);

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

    let meshTriangles = 0;
    const groups = mesh.indices || [];
    groups.forEach((group, groupIndex) => {
      const groupName = sanitizeName(group?.name, `area_${groupIndex}`);
      lines.push(`g ${objectName}_${groupName}`, `usemtl ${groupName}`);
      const faces = group?.faces || [];
      for (let i = 0; i + 2 < faces.length; i += 3) {
        const local = [faces[i], faces[i + 1], faces[i + 2]];
        if (local.some((index) => !Number.isInteger(index) || index < 0 || index >= vertexCount)) {
          throw new Error(`Mesh ${meshIndex}, group ${groupIndex} contains an invalid triangle index`);
        }
        const tokens = local.map((index) => {
          const v = vertexOffset + index + 1;
          const vt = uvOffset + index + 1;
          if (hasNormals) {
            const vn = normalOffset + index + 1;
            return `${v}/${vt}/${vn}`;
          }
          return `${v}/${vt}`;
        });
        lines.push(`f ${tokens.join(" ")}`);
        meshTriangles += 1;
      }
    });

    meshSummaries.push({
      index: meshIndex,
      name: mesh?.name || objectName,
      vertices: vertexCount,
      triangles: meshTriangles,
      groups: groups.length,
      hasNormals,
      hasUv0: true
    });

    vertexOffset += vertexCount;
    uvOffset += uvCount;
    if (hasNormals) normalOffset += normalCount;
    totalVertices += vertexCount;
    totalTriangles += meshTriangles;
  });

  if (!totalVertices || !totalTriangles) {
    throw new Error("GR2 conversion produced no renderable triangles");
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${lines.join("\n")}\n`, "utf8");

  const summary = {
    source: path.resolve(inputPath),
    output: path.resolve(outputPath),
    grannyFileFormatRevision: graph.grannyFileFormatRevision,
    grannyFileSource: graph.grannyFileSource,
    meshes: meshSummaries,
    totalVertices,
    totalTriangles
  };
  if (summaryPath) {
    fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  }
  console.log(`Converted GR2 -> OBJ: ${outputPath}`);
  console.log(`Meshes=${meshSummaries.length} vertices=${totalVertices} triangles=${totalTriangles}`);
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

function usage() {
  console.log("Usage:");
  console.log("  node convert_eve_asset.mjs gr2-to-obj <input.gr2> <output.obj> [summary.json]");
  console.log("  node convert_eve_asset.mjs dds-to-png <input.dds> <output.png>");
  console.log("  node convert_eve_asset.mjs dds-to-environment-png <input.dds> <output.png>");
}

try {
  const [command, inputPath, outputPath, summaryPath] = process.argv.slice(2);
  if (!command || !inputPath || !outputPath) {
    usage();
    process.exitCode = 2;
  } else if (command === "gr2-to-obj") {
    writeObj(inputPath, outputPath, summaryPath);
  } else if (command === "dds-to-png") {
    writePngFromDds(inputPath, outputPath);
  } else if (command === "dds-to-environment-png") {
    writeEnvironmentPngFromDds(inputPath, outputPath);
  } else {
    usage();
    fail(`Unknown command ${command}`);
  }
} catch (error) {
  fail(error?.stack || String(error));
}

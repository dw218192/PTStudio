"""Download LTC reference data from Heitz 2016 and generate C++ header."""
import urllib.request
import re
import sys
from pathlib import Path


def fetch_url(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def extract_array(text: str, name: str) -> list[float]:
    """Extract a C-style float array from text."""
    # Find the array declaration and its contents (handles any type before name)
    pattern = rf'{name}\s*\[.*?\]\s*=\s*\{{(.*?)\}}\s*;'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find array '{name}' in source")
    content = match.group(1)
    # Extract all float values (including nested braces for mat33)
    values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?', content)
    return [float(v.rstrip('f')) for v in values]


def main():
    if len(sys.argv) < 2:
        print("Usage: fetch_ltc_data.py <output_header_path>")
        sys.exit(1)

    output_path = Path(sys.argv[1])

    # Download the .inc file with raw LTC data
    url = "https://raw.githubusercontent.com/selfshadow/ltc_code/master/fit/results/ltc.inc"
    print(f"Downloading {url}...")
    text = fetch_url(url)
    print(f"Downloaded {len(text)} bytes")

    # Extract arrays
    # tabM: 64*64 entries, each with 9 floats (3x3 matrix) but stored as 4 independent params
    # Actually, ltc.inc stores tabM as 9 floats per entry and tabAmplitude separately
    # Let's check the format by looking at the structure

    # The .inc file has:
    # - float tabM[size*9] - the 3x3 M^-1 matrices
    # - float tabAmplitude[size] - the amplitude/norm

    tab_m = extract_array(text, "tabM")
    print(f"tabM: {len(tab_m)} values ({len(tab_m)//9} matrices)")

    tab_amp = extract_array(text, "tabMagnitude")
    print(f"tabMagnitude: {len(tab_amp)} values")

    n = 64  # LUT size
    assert len(tab_m) == n * n * 9, f"Expected {n*n*9} values in tabM, got {len(tab_m)}"
    assert len(tab_amp) == n * n, f"Expected {n*n} values in tabAmplitude, got {len(tab_amp)}"

    # Convert tabM (9 floats per entry) to 4 floats per texel
    # The 3x3 matrix has structure: | a  0  b |
    #                                | 0  c  0 |
    #                                | d  0  1 |
    # But we need to check the actual layout from the reference...
    # From the reference code, the matrix is stored row-major as:
    # m[0] m[1] m[2]   =   a  b  c
    # m[3] m[4] m[5]       d  1  e
    # m[6] m[7] m[8]       f  g  h
    #
    # Actually from Heitz's ltc.h, the matrix has this structure:
    #   | a  0  b |
    #   | 0  c  0 |
    #   | d  0  e |
    # Which means only 5 independent values: a, b, c, d, e
    # But the reference WebGL code packs into RGBA as (a, b, c, d) with e=1
    # Let me check the .js file to see how it's packed...

    # From the standard LTC implementation, the RGBA packing is:
    # Channel R: m[0][0] (a)
    # Channel G: m[0][2] (b)
    # Channel B: m[2][0] (d)
    # Channel A: m[2][2] (e) -- but this is always close to 1.0... hmm
    #
    # Actually, let me look at the Three.js LTC implementation which is well-documented
    # Three.js stores: mat[0], mat[2], mat[4], mat[6] as RGBA
    # where the 3x3 matrix stored row-major is:
    # [0] [1] [2]     a  0  b
    # [3] [4] [5]  =  0  c  0
    # [6] [7] [8]     d  0  e
    #
    # So the 4 packed values are: a (m00), b (m02), c (m11), d (m20)
    # And m22 (e) is reconstructed or stored as the alpha.
    #
    # The standard packing for a 4-channel texture is:
    # R = m00 (a), G = m02 (b), B = m20 (d), A = m22 (e)
    # And m11 (c) = 1 always? No...
    #
    # Let me just store all 4 unique non-trivial values.
    # From the reference WebGL shader:
    #   mat3 Minv = mat3(
    #     vec3(t1.x, 0, t1.y),
    #     vec3(  0,  1,    0),
    #     vec3(t1.z, 0, t1.w));
    # So RGBA = (m00, m02, m20, m22) and m11=1.

    # tabM stores the raw fitted M matrix in column-major order (GLM convention).
    # We need M^-1 normalized by M^-1[1][1], matching the reference packTab():
    #   invM = inverse(m); invM /= invM[1][1];
    #   tex1 = (invM[0][0], invM[0][2], invM[2][0], invM[2][2])  (GLM col-major)
    # In mathematical (row,col) notation: RGBA = (m00, m20, m02, m22) of normalized M^-1.
    #
    # The matrix has sparsity: M = | a 0 b | / | 0 c 0 | / | d 0 e |
    # (zero-forced during fitting). Inverse is straightforward.

    def invert_and_normalize(raw):
        """Invert a sparse 3x3 matrix from column-major storage and normalize by [1][1]."""
        # Column-major: raw[0:3]=col0, raw[3:6]=col1, raw[6:9]=col2
        # Mathematical matrix (row, col):
        #   M[0][0]=raw[0], M[0][1]=raw[3], M[0][2]=raw[6]
        #   M[1][0]=raw[1], M[1][1]=raw[4], M[1][2]=raw[7]
        #   M[2][0]=raw[2], M[2][1]=raw[5], M[2][2]=raw[8]
        a, c, b = raw[0], raw[4], raw[6]  # m00, m11, m02
        d, e = raw[2], raw[8]              # m20, m22

        # M = | a  0  b |    Inverse = (1/det) * | ce   0  -bc |
        #     | 0  c  0 |                         |  0  ae-bd  0 |
        #     | d  0  e |                         | -cd  0   ac |
        det = c * (a * e - b * d)
        if abs(det) < 1e-30:
            return [1.0, 0.0, 0.0, 1.0]

        inv_det = 1.0 / det
        inv00 = c * e * inv_det
        inv02 = -b * c * inv_det
        inv11 = (a * e - b * d) * inv_det
        inv20 = -c * d * inv_det
        inv22 = a * c * inv_det

        # Normalize by inv[1][1]
        if abs(inv11) < 1e-30:
            return [1.0, 0.0, 0.0, 1.0]
        inv00 /= inv11
        inv02 /= inv11
        inv20 /= inv11
        inv22 /= inv11

        # RGBA = (m00, m20, m02, m22) matching reference GLM packing
        return [inv00, inv20, inv02, inv22]

    ltc_mat = []  # 4 floats per texel
    for i in range(n * n):
        base = i * 9
        raw = tab_m[base:base + 9]
        ltc_mat.extend(invert_and_normalize(raw))

    # Also fetch the Fresnel data from ltc.js or ltc.h
    # The .inc only has tabAmplitude (scalar norm), but we also need Fresnel
    # Let's try fetching ltc.h which may have both
    url_h = "https://raw.githubusercontent.com/selfshadow/ltc_code/master/fit/results/ltc.h"
    print(f"\nDownloading {url_h}...")
    text_h = fetch_url(url_h)
    print(f"Downloaded {len(text_h)} bytes")

    # ltc.h might have tabMagFresnel or similar
    # Let's check what arrays are in it
    array_names = re.findall(r'(?:float|double)\s+(\w+)\s*\[', text_h)
    print(f"Arrays found in ltc.h: {array_names}")

    # Try to find a Fresnel array
    has_fresnel = False
    tab_fresnel = None
    for name in array_names:
        if 'fresnel' in name.lower() or 'mag' in name.lower():
            tab_fresnel = extract_array(text_h, name)
            print(f"{name}: {len(tab_fresnel)} values")
            has_fresnel = True

    if not has_fresnel:
        # Check the .js file
        url_js = "https://raw.githubusercontent.com/selfshadow/ltc_code/master/fit/results/ltc.js"
        print(f"\nDownloading {url_js}...")
        text_js = fetch_url(url_js)
        print(f"Downloaded {len(text_js)} bytes")

        # Look for g_ltc_2 or similar
        array_names_js = re.findall(r'var\s+(\w+)\s*=\s*\[', text_js)
        print(f"Arrays found in ltc.js: {array_names_js}")

        for name in array_names_js:
            if '2' in name or 'amp' in name.lower() or 'fresnel' in name.lower():
                # Extract JS array
                pattern = rf'{name}\s*=\s*\[(.*?)\]\s*;'
                match = re.search(pattern, text_js, re.DOTALL)
                if match:
                    values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', match.group(1))
                    tab_fresnel = [float(v) for v in values]
                    print(f"{name}: {len(tab_fresnel)} values")
                    has_fresnel = True
                    break

    # Build amplitude LUT (2 floats per texel: magnitude, fresnel)
    ltc_amp = []
    if has_fresnel and tab_fresnel is not None:
        # The JS g_ltc_2 is RGBA with 4 floats per texel
        floats_per_texel = len(tab_fresnel) // (n * n)
        print(f"Fresnel data: {floats_per_texel} floats per texel")
        for i in range(n * n):
            if floats_per_texel == 4:
                base = i * 4
                ltc_amp.extend([tab_fresnel[base], tab_fresnel[base + 1]])
            elif floats_per_texel == 2:
                base = i * 2
                ltc_amp.extend([tab_fresnel[base], tab_fresnel[base + 1]])
            else:
                ltc_amp.extend([tab_fresnel[i], 1.0])
    else:
        # Fall back: use tabAmplitude as magnitude, fresnel=1
        for i in range(n * n):
            ltc_amp.extend([tab_amp[i], 1.0])

    print(f"\nFinal data sizes:")
    print(f"  ltc_mat: {len(ltc_mat)} floats ({len(ltc_mat) * 4} bytes)")
    print(f"  ltc_amp: {len(ltc_amp)} floats ({len(ltc_amp) * 4} bytes)")

    # Write C++ header
    def format_floats(values, per_line=8):
        lines = []
        for i in range(0, len(values), per_line):
            chunk = values[i:i + per_line]
            line = ", ".join(f"{v:.8e}f" for v in chunk)
            lines.append(f"    {line},")
        return "\n".join(lines)

    header = f"""#pragma once
// LTC (Linearly Transformed Cosines) lookup tables for GGX BRDF.
// Reference: Heitz et al., "Real-Time Polygonal-Light Shading with LTC", SIGGRAPH 2016
// Source: https://github.com/selfshadow/ltc_code
//
// Generated by _tools/fetch_ltc_data.py — do not edit manually.

#include <cstddef>

namespace pts::rendering {{

static constexpr size_t k_ltc_size = {n};

// {n}x{n} RGBA — M^(-1) matrix parameters (inverted and normalized from raw fit).
// Each texel stores (m00, m20, m02, m22) of the 3x3 inverse LTC matrix.
// The matrix has structure: | m00   0  m02 |
//                           |   0   1    0 |
//                           | m20   0  m22 |
// Note: RGBA channel order is (m00, m20, m02, m22) to match the reference
// GLSL mat3 column-major constructor convention.
// Indexed as [y * {n} + x] where x = roughness, y = sqrt(1 - cos_theta).
static constexpr float k_ltc_mat[] = {{
{format_floats(ltc_mat)}
}};

// {n}x{n} RG — Fresnel-weighted amplitude.
// Each texel stores (magnitude, fresnel_term).
static constexpr float k_ltc_amp[] = {{
{format_floats(ltc_amp)}
}};

}}  // namespace pts::rendering
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header)
    print(f"\nWrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

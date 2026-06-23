/**
 * heic.ts — client-side HEIC/HEIF → JPEG conversion.
 *
 * iPhones export photos as HEIC. Most browsers (Chrome/Firefox) can't render
 * HEIC in an <img>, and the backend's Pillow decoder can't read it without the
 * pillow-heif plugin — so HEIC uploads would fail to preview AND get rejected
 * as "couldn't read that photo". We detect HEIC files in the browser and
 * convert them to JPEG before upload, so the previews and the reconstruction
 * pipeline both receive a format they support. No backend changes needed.
 */

// HEIF/HEIC ISO-BMFF `ftyp` major / compatible brands.
const HEIC_BRANDS = new Set([
  'heic', 'heix', 'heim', 'heis',
  'hevc', 'hevx', 'hevm', 'hevs',
  'mif1', 'msf1', 'mif2',
]);

/**
 * True if `file` is a HEIC/HEIF image. Checks the extension and MIME type first
 * (covers the common iPhone case), then falls back to sniffing the ISO-BMFF
 * `ftyp` brand from the file header for files with a generic name or type.
 */
export async function isHeic(file: File): Promise<boolean> {
  const name = file.name.toLowerCase();
  if (name.endsWith('.heic') || name.endsWith('.heif')) return true;

  const type = file.type.toLowerCase();
  if (type === 'image/heic' || type === 'image/heif') return true;

  try {
    const header = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    // Bytes 4..8 are the box type; for HEIF this is "ftyp".
    const boxType = String.fromCharCode(...header.subarray(4, 8));
    if (boxType !== 'ftyp') return false;
    const brand = String.fromCharCode(...header.subarray(8, 12));
    return HEIC_BRANDS.has(brand);
  } catch {
    return false;
  }
}

/**
 * Convert a HEIC/HEIF File to a JPEG File. The heavy libheif/WASM bundle is
 * imported lazily on first use so it never weighs down the initial page load.
 */
export async function convertHeicToJpeg(
  file: File,
  quality = 0.92,
): Promise<File> {
  const { default: heic2any } = await import('heic2any');
  const out = await heic2any({ blob: file, toType: 'image/jpeg', quality });
  const blob = Array.isArray(out) ? out[0] : out;
  const jpegName = file.name.replace(/\.(heic|heif)$/i, '') + '.jpg';
  return new File([blob], jpegName, {
    type: 'image/jpeg',
    lastModified: file.lastModified,
  });
}

/**
 * Convert any HEIC/HEIF files in a batch to JPEG, passing everything else
 * through unchanged and preserving order. Returns the resulting files plus the
 * names of any HEIC files that failed to convert (skipped), so the caller can
 * show a friendly message.
 */
export async function convertHeicFiles(
  files: File[],
): Promise<{ files: File[]; failed: string[] }> {
  const out: File[] = [];
  const failed: string[] = [];
  for (const file of files) {
    if (await isHeic(file)) {
      try {
        out.push(await convertHeicToJpeg(file));
      } catch {
        failed.push(file.name);
      }
    } else {
      out.push(file);
    }
  }
  return { files: out, failed };
}

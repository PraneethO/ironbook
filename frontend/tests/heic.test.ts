import { describe, expect, it, vi } from 'vitest';

// Stub the heavy libheif/WASM bundle so convertHeicFiles is testable in jsdom.
const heic2any = vi.fn(
  async () => new Blob([new Uint8Array([0xff, 0xd8, 0xff])], { type: 'image/jpeg' }),
);
vi.mock('heic2any', () => ({ default: heic2any }));

import { isHeic, convertHeicFiles } from '../src/lib/heic';

function fileWith(name: string, type = '', bytes = new Uint8Array([1, 2, 3])): File {
  return new File([bytes], name, { type });
}

/** A 12-byte ISO-BMFF header with the given ftyp brand. */
function heicHeaderFile(name: string, brand: string, type = ''): File {
  const bytes = new Uint8Array(12);
  const write = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) bytes[off + i] = s.charCodeAt(i);
  };
  write(4, 'ftyp');
  write(8, brand);
  return fileWith(name, type, bytes);
}

describe('isHeic', () => {
  it('detects by extension (case-insensitive)', async () => {
    expect(await isHeic(fileWith('IMG_0001.HEIC'))).toBe(true);
    expect(await isHeic(fileWith('photo.heif'))).toBe(true);
  });

  it('detects by MIME type', async () => {
    expect(await isHeic(fileWith('blob', 'image/heic'))).toBe(true);
  });

  it('detects by ftyp brand for generically-named files', async () => {
    expect(await isHeic(heicHeaderFile('blob', 'heic'))).toBe(true);
    expect(await isHeic(heicHeaderFile('blob', 'mif1'))).toBe(true);
  });

  it('returns false for ordinary JPEG/PNG', async () => {
    expect(await isHeic(fileWith('a.jpg', 'image/jpeg'))).toBe(false);
    expect(await isHeic(fileWith('a.png', 'image/png'))).toBe(false);
  });
});

describe('convertHeicFiles', () => {
  it('converts HEIC to JPEG and passes other files through, preserving order', async () => {
    const jpg = fileWith('keep.jpg', 'image/jpeg');
    const heic = fileWith('IMG_0001.heic', 'image/heic');

    const { files, failed } = await convertHeicFiles([jpg, heic]);

    expect(failed).toEqual([]);
    expect(files).toHaveLength(2);
    expect(files[0]).toBe(jpg); // untouched
    expect(files[1].name).toBe('IMG_0001.jpg');
    expect(files[1].type).toBe('image/jpeg');
    expect(heic2any).toHaveBeenCalledTimes(1);
  });

  it('reports HEIC files that fail to convert and skips them', async () => {
    heic2any.mockRejectedValueOnce(new Error('decode failed'));
    const heic = fileWith('broken.heic', 'image/heic');

    const { files, failed } = await convertHeicFiles([heic]);

    expect(files).toEqual([]);
    expect(failed).toEqual(['broken.heic']);
  });
});

import '@testing-library/jest-dom/vitest';

// jsdom doesn't implement object URLs; stub them for preview/screenshot code.
if (typeof URL.createObjectURL !== 'function') {
  URL.createObjectURL = () => 'blob:mock';
}
if (typeof URL.revokeObjectURL !== 'function') {
  URL.revokeObjectURL = () => undefined;
}

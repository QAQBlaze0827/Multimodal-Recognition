# opencc-js 1.4.2

Unmodified browser UMD bundle: `dist/umd/cn2t.js` from the published package.
Only the simplified-to-traditional bundle is included, not the full package.

- Project: https://github.com/nk2028/opencc-js
- Package: https://registry.npmjs.org/opencc-js/1.4.2
- Archive: https://registry.npmjs.org/opencc-js/-/opencc-js-1.4.2.tgz
- Archive integrity: `sha512-EdcspgdROiCCZDM+jZR6Pa670xW+EW2QhctPBQ8gRjg6lrx4IHEl+2s+kt5izk/TkxVRgdWuFgaGkaEdIUfdaw==`
- `cn2t.js` SHA-256: `d66d920069ab526db08c6f2bc15fcf6feb7d07de3cf98e7e27c0aaf7d136baee`
- `cn2t.js` size: 1,100,656 bytes.
- Licenses: MIT for opencc-js; Apache-2.0 for the bundled dictionary data.
  Preserve `LICENSE`, `LICENSES/Apache-2.0.txt`, and `THIRD_PARTY_LICENSES.md`.

The application uses `OpenCC.Converter({ from: 'cn', to: 'tw' })` for display
only, without the additional `twp` regional vocabulary substitutions. The
browser loads this local file; no transcript is sent to a conversion service
and no external dictionary request is needed. Conversion is not speech-error
correction and can still be ambiguous for some words.

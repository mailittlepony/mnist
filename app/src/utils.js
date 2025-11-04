/*
 * utils.js
 * Copyright (C) 2025 mailitg <mailitg@maili-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

export function preprocess(pctx) {
    // downscale
    const off = document.createElement("canvas");
    off.width = 28; off.height = 28;
    const offCtx = off.getContext("2d");
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = "high";
    offCtx.drawImage(draw, 0, 0, 28, 28);

    // preview 
    pctx.imageSmoothingEnabled = false;
    pctx.clearRect(0, 0, preview.width, preview.height);
    pctx.drawImage(off, 0, 0, 112, 112);

    // read pixels
    const { data } = offCtx.getImageData(0, 0, 28, 28);
    const input = new Float32Array(28*28);
    for (let i=0;i<28*28;i++) {
        const r = data[i*4+0], g = data[i*4+1], b = data[i*4+2];
        const gray = (r+g+b)/3;
        input[i] = (gray*2/255) - 1;
    }
    return input;
}

/*
 * utils.js
 * Copyright (C) 2025 mailitg <mailitg@maili-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

export function preprocess(pctx, draw) {
    // downscale
    const off = document.createElement("canvas");
    off.width = 28; off.height = 28;
    const offCtx = off.getContext("2d");
    offCtx.imageSmoothingEnabled = true;
    offCtx.imageSmoothingQuality = "high";
    offCtx.drawImage(draw, 0, 0, 28, 28);

    // preview 
    const pv = pctx.canvas;
    pctx.imageSmoothingEnabled = false;
    pctx.clearRect(0, 0, pv.width, pv.height);
    pctx.drawImage(off, 0, 0, pv.width, pv.height);

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

// Convert logits to probabilities
export function softmax(logits) {
    const arr = Array.from(logits || []); 
    const max = Math.max(...arr);        
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map(v => v / sum);
}

export function argmax(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
}


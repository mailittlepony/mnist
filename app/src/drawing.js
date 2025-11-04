/*
 * drawing.js
 * Copyright (C) 2025 mailitg <mailitg@maili-mba.local>
 *
 * Distributed under terms of the MIT license.
 */


export function setupDrawing(drawCanvas, { bg = "black", penColor = "white", penWidth = 20 } = {}) {
    const ctx = drawCanvas.getContext("2d");

    function resetCanvas() {
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    }

    let drawing = false;
    let last = null;
    let tool = "pen"; // 'pen' | 'erase'

    const pen = { color: penColor, width: penWidth, cap: "round", join: "round" };

    function strokeStyle() {
        if (tool === "erase") {
            // "erase" by painting background color
            ctx.strokeStyle = bg;
        } else {
            ctx.strokeStyle = pen.color;
        }
        ctx.lineWidth = pen.width;
        ctx.lineCap = pen.cap;
        ctx.lineJoin = pen.join;
    }

    function startDraw(x, y) { drawing = true; last = { x, y }; }
    function moveDraw(x, y) {
        if (!drawing) return;
        strokeStyle();
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(x, y);
        ctx.stroke();
        last = { x, y };
    }
    function endDraw() { drawing = false; last = null; }

    // mouse
    drawCanvas.addEventListener("mousedown", e => {
        const r = drawCanvas.getBoundingClientRect();
        startDraw(e.clientX - r.left, e.clientY - r.top);
    });
    drawCanvas.addEventListener("mousemove", e => {
        const r = drawCanvas.getBoundingClientRect();
        moveDraw(e.clientX - r.left, e.clientY - r.top);
    });
    window.addEventListener("mouseup", endDraw);

    // touch
    drawCanvas.addEventListener("touchstart", e => {
        const t = e.touches[0];
        const r = drawCanvas.getBoundingClientRect();
        startDraw(t.clientX - r.left, t.clientY - r.top);
        e.preventDefault();
    }, { passive: false });
    drawCanvas.addEventListener("touchmove", e => {
        const t = e.touches[0];
        const r = drawCanvas.getBoundingClientRect();
        moveDraw(t.clientX - r.left, t.clientY - r.top);
        e.preventDefault();
    }, { passive: false });
    window.addEventListener("touchend", endDraw);

    // init
    resetCanvas();

    return {
        ctx,
        resetCanvas,
        setTool(next) { tool = next; }, // 'pen' or 'erase'
        get tool() { return tool; }
    };
}


/*
 * main.js
 * Copyright (C) 2025 mailitg <mailitg@maili-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

import { makeGuess, setupInference} from "./inference";
import { preprocess } from "./utils";

const draw = document.getElementById("draw");
const clearBtn = document.getElementById("clear");
const preview = document.getElementById("preview");
const pred = document.getElementById("pred");
const conf = document.getElementById("conf");
const runBtn = document.getElementById("run");
const statusText = document.getElementById("status");
const modelSelect = document.getElementById("model");
const compileTime = document.getElementById("compileTime");
const inferTime = document.getElementById("inferTime");

const ctx = draw.getContext("2d");
const pctx = preview.getContext("2d");

function resetCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, draw.width, draw.height);
}
resetCanvas();

let drawing = false;
let last = null;
const pen = { color: "white", width: 20, cap: "round", join: "round" };

function startDraw(x, y) { drawing = true; last = {x,y}; }
function moveDraw(x, y) {
    if (!drawing) return;
    ctx.strokeStyle = pen.color;
    ctx.lineWidth = pen.width;
    ctx.lineCap = pen.cap;
    ctx.lineJoin = pen.join;
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    last = {x,y};
}
function endDraw() { drawing = false; last = null; }

draw.addEventListener("mousedown", e => {
    const r = draw.getBoundingClientRect();
    startDraw(e.clientX - r.left, e.clientY - r.top);
});
draw.addEventListener("mousemove", e => {
    const r = draw.getBoundingClientRect();
    moveDraw(e.clientX - r.left, e.clientY - r.top);
});
window.addEventListener("mouseup", endDraw);

// touch
draw.addEventListener("touchstart", e => {
    const t = e.touches[0];
    const r = draw.getBoundingClientRect();
    startDraw(t.clientX - r.left, t.clientY - r.top);
    e.preventDefault();
}, {passive:false});
draw.addEventListener("touchmove", e => {
    const t = e.touches[0];
    const r = draw.getBoundingClientRect();
    moveDraw(t.clientX - r.left, t.clientY - r.top);
    e.preventDefault();
}, {passive:false});
window.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", () => {
    resetCanvas();
    pctx.clearRect(0,0,preview.width, preview.height);
    pred.textContent = "—";
    conf.textContent = "—";
    compileTime.textContent = "—";
    inferTime.textContent = "—";
});

statusText.textContent = "draw something and press Run";
runBtn.addEventListener("click", () => {
    alert("Next step will run the model. For now, just testing UI.");
});

// test it via Run button
runBtn.addEventListener("click", () => {
    const input = preprocess(pctx);
    console.log("input[0..9] = ", input.slice(0,10));
    alert("Preprocess done (check console). Next step will run the model.");
}, { once: true });




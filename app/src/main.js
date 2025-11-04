/*
 * main.js
 * Copyright (C) 2025 mailitg <mailitg@maili-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

import Chart from 'chart.js/auto';
import { makeGuess, setupInference } from "./inference";
import { preprocess, softmax, argmax } from "./utils";
import { setupDrawing } from "./drawing";

const draw = document.getElementById("draw");
const clearBtn = document.getElementById("clear");
const preview = document.getElementById("preview");
const pred = document.getElementById("pred");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const probsCanvas = document.getElementById('probsChart');

const pctx = preview.getContext("2d");
const drawing = setupDrawing(draw, { bg: "black", penColor: "white", penWidth: 20 });
const ctx = probsCanvas.getContext('2d');

const setStatus = (t) => (statusEl.textContent = t);
const setTimer = (t) => (timerEl.textContent = t ?? "—");
const setPred = (n) => (pred.textContent = n === null || n === undefined ? "" : n.toString());


function resetUI() {
    pctx.clearRect(0, 0, preview.width, preview.height);
    setPred("");
    updateProbChart(probChart, new Array(10).fill(0));
}

let isReady = false;
function setReady(ready, loadMs) {
    isReady = ready;
    runBtn.disabled = !ready;
    setStatus(ready ? "ready" : "loading…");
    if (typeof loadMs === "number") setTimer(`${loadMs.toFixed(0)} ms to load`);
}

(async function init() {
    try {
        setReady(false);
        const t0 = performance.now();
        await setupInference();
        const t1 = performance.now();
        setReady(true, t1 - t0);
    } catch (e) {
        setStatus("error");
        console.error(e);
    }
})();

const DIGITS = Array.from({ length: 10 }, (_, i) => i.toString());

function createProbChart(ctx) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: DIGITS,
            datasets: [{ label: 'Probability', data: new Array(10).fill(0), borderWidth: 1 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 180 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => {
                            const v = item.raw ?? 0;
                            return v;
                        }
                    }
                }
            },
            scales: { y: { min: 0, max: 1} }
        }
    });
}

const probChart = createProbChart(ctx);

function updateProbChart(chart, probs = []) {
    chart.data.datasets[0].data = probs;

    const k = argmax(probs);
    chart.data.datasets[0].backgroundColor = probs.map((_, i) =>
        i === k ? "rgba(99,102,241,0.6)" : "rgba(148,163,184,0.3)"
    );
    chart.data.datasets[0].borderColor = probs.map((_, i) =>
        i === k ? "rgba(99,102,241,1)" : "rgba(148,163,184,0.4)"
    );

    chart.update();
}

clearBtn.addEventListener("click", () => {
    drawing.resetCanvas();
    resetUI();
});

runBtn.addEventListener("click", async () => {
    if (!isReady) return; // guard
    const input = preprocess(pctx, draw);

    const result = await Promise.resolve(makeGuess(input));

    const logits = result?.output?.data ?? result?.output ?? result ?? [];
    const probs = softmax(logits);
    const predIdx = argmax(probs);                                  

    setPred(Number.isFinite(predIdx) ? predIdx : "");
    updateProbChart(probChart, probs);
});


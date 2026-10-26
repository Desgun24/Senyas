"use strict";

const video = document.getElementById("preview");
const startButton = document.getElementById("start");
const recordButton = document.getElementById("record");
const stopButton = document.getElementById("stop");
const status = document.getElementById("status");
const device = document.getElementById("device");
const countdown = document.getElementById("countdown");
const parentOrigin = new URL(document.referrer || window.location.href).origin;
let stream = null;
let starting = false;
let recording = false;
let pendingCapture = null;
let context = null;
let runToken = 0;
let durationSeconds = 4;
let frameCount = 32;
let lastHeight = 0;

function post(type, data = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, parentOrigin);
}
function setHeight() {
  const height = Math.ceil(document.body.getBoundingClientRect().height) + 8;
  if (height !== lastHeight) {
    lastHeight = height;
    post("streamlit:setFrameHeight", { height });
  }
}
function setStatus(message) { status.textContent = message; setHeight(); }
function controls() {
  startButton.disabled = starting || Boolean(stream);
  recordButton.disabled = !stream || starting || recording || Boolean(pendingCapture) || !context;
  stopButton.disabled = !stream && !starting;
  device.disabled = starting || recording || Boolean(pendingCapture);
  document.getElementById("placeholder").hidden = Boolean(stream);
}
function cancelRecording() {
  runToken += 1;
  recording = false;
  countdown.hidden = true;
  controls();
}
function stopCamera(message = "Camera stopped.") {
  cancelRecording();
  starting = false;
  const previous = stream;
  stream = null;
  if (previous) previous.getTracks().forEach(track => track.stop());
  video.pause();
  video.srcObject = null;
  controls();
  setStatus(message);
}
function cameraError(error) {
  if (error.name === "NotAllowedError" || error.name === "SecurityError") {
    return "Camera access was blocked. Allow camera access for this page, then press Start camera again. If the embedded browser blocks it, open this same local address in Edge or Chrome.";
  }
  if (error.name === "NotFoundError") return "No camera was found. Connect a webcam and try again.";
  if (error.name === "NotReadableError") return "The camera is busy. Close other apps using it, then try again.";
  return "The camera could not start: " + (error.message || error.name);
}
async function startCamera() {
  if (starting || stream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Camera access is unavailable here. Open the app at localhost or 127.0.0.1 in Edge or Chrome.");
    return;
  }
  const token = ++runToken;
  starting = true;
  controls();
  setStatus("Waiting for camera permission…");
  try {
    const constraints = { width: { ideal: 640 }, height: { ideal: 360 }, aspectRatio: { ideal: 16 / 9 }, frameRate: { ideal: 30, max: 30 } };
    if (device.value) constraints.deviceId = { exact: device.value };
    else constraints.facingMode = "user";
    const acquired = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
    if (token !== runToken) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired;
    const cameraTrack = stream.getVideoTracks()[0];
    cameraTrack.addEventListener("ended", () => {
      if (stream === acquired) stopCamera("The camera disconnected. Start it again to continue.");
    });
    video.srcObject = stream;
    await video.play();
    if (token !== runToken) return;
    if (cameraTrack.readyState !== "live") throw new Error("The camera stopped before the preview was ready. Please start it again.");
    const selectedDevice = cameraTrack.getSettings().deviceId;
    const devices = await navigator.mediaDevices.enumerateDevices();
    if (token !== runToken) return;
    device.replaceChildren();
    for (const [index, camera] of devices.filter(item => item.kind === "videoinput").entries()) {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || "Camera " + (index + 1);
      option.selected = camera.deviceId === selectedDevice;
      device.appendChild(option);
    }
    document.getElementById("device-label").hidden = device.options.length < 2;
    starting = false;
    controls();
    const cameraFps = cameraTrack.getSettings().frameRate;
    setStatus("Camera ready" + (cameraFps ? " · device reports " + Number(cameraFps).toFixed(1) + " FPS" : "") + ". Press Record 4-second sign when you are ready.");
  } catch (error) {
    if (token === runToken) stopCamera(cameraError(error));
  }
}

const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
async function recordSign() {
  if (recordButton.disabled) return;
  const token = ++runToken;
  const captureContext = context;
  recording = true;
  controls();
  try {
    for (let remaining = 3; remaining > 0; remaining -= 1) {
      countdown.hidden = false;
      countdown.firstElementChild.textContent = String(remaining);
      setStatus("Get ready: " + remaining);
      await delay(1000);
      if (token !== runToken) return;
    }
    countdown.hidden = true;
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 360;
    const drawing = canvas.getContext("2d", { alpha: false });
    const frames = [];
    const timestamps = [];
    const started = performance.now();
    for (let index = 0; index < frameCount; index += 1) {
      const target = started + index * durationSeconds * 1000 / (frameCount - 1);
      // Follow presentation callbacks directly at high rates. Delaying before
      // requesting a callback can miss every second frame on some browsers.
      if (frameCount > 32 && typeof video.requestVideoFrameCallback === "function") {
        await Promise.race([
          new Promise(resolve => video.requestVideoFrameCallback(resolve)),
          delay(1000).then(() => { throw new Error("No new camera frame arrived. Please record again."); })
        ]);
      } else {
        await delay(Math.max(0, target - performance.now()));
      }
      if (token !== runToken) return;
      // Stop the development recording by elapsed time, not by a promised
      // frame rate. Slow cameras yield fewer samples and keep truthful timing.
      if (frameCount > 32 && frames.length >= 2 && performance.now() - started >= durationSeconds * 1000) break;
      if (document.hidden || !stream || video.readyState < 2 || stream.getVideoTracks()[0].readyState !== "live") {
        throw new Error("The camera paused. Keep this tab active and record again.");
      }
      // Match the preview's centered 16:9 crop, without its optional CSS mirror.
      const scale = Math.max(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
      const width = video.videoWidth * scale;
      const height = video.videoHeight * scale;
      drawing.drawImage(video, (canvas.width - width) / 2, (canvas.height - height) / 2, width, height);
      timestamps.push((performance.now() - started) / 1000);
      frames.push(canvas.toDataURL("image/jpeg", 0.8).split(",")[1]);
      setStatus("Recording… " + Math.max(0, durationSeconds - (performance.now() - started) / 1000).toFixed(1) + " seconds left");
    }
    pendingCapture = crypto.randomUUID();
    recording = false;
    controls();
    setStatus("Analyzing your sign…");
    post("streamlit:setComponentValue", {
      dataType: "json",
      value: { capture_id: pendingCapture, context: captureContext, frames, timestamps }
    });
  } catch (error) {
    if (token !== runToken) return;
    recording = false;
    countdown.hidden = true;
    controls();
    setStatus(error.message || "The recording could not be completed. Please try again.");
  }
}

window.addEventListener("message", event => {
  if (event.source !== window.parent || event.origin !== parentOrigin || event.data?.type !== "streamlit:render") return;
  const args = event.data.args || {};
  if (context !== null && context !== args.context) {
    cancelRecording();
    pendingCapture = null;
    post("streamlit:setComponentValue", { dataType: "json", value: null });
    setStatus(stream ? "Target or model changed. Record a new signing attempt." : "Start the camera to record this sign.");
  }
  context = args.context;
  durationSeconds = args.duration_seconds || 4;
  frameCount = args.frame_count || 32;
  if (pendingCapture && args.processed_capture_id === pendingCapture) {
    pendingCapture = null;
    post("streamlit:setComponentValue", { dataType: "json", value: null });
    setStatus(stream ? "Ready — you can record another sign." : "Camera stopped.");
  }
  controls();
  setHeight();
});
startButton.addEventListener("click", startCamera);
recordButton.addEventListener("click", recordSign);
stopButton.addEventListener("click", () => stopCamera());
device.addEventListener("change", () => { stopCamera(); startCamera(); });
document.getElementById("mirror").addEventListener("change", event => video.classList.toggle("mirrored", event.target.checked));
document.addEventListener("visibilitychange", () => { if (document.hidden) stopCamera("Camera stopped while this tab was hidden."); });
window.addEventListener("pagehide", () => stopCamera());
new ResizeObserver(setHeight).observe(document.body);
post("streamlit:componentReady", { apiVersion: 1 });
setHeight();

#!/usr/bin/env python3
"""Rover drive server — fullscreen camera, circular D-pad, line following,
tour and go-to-goal (learned policy).

Run on the Pi:
    python3 rover_server.py
Then on your phone (same Wi-Fi):
    http://<your-pi's-address>:8001

The camera video fills the screen. A circular D-pad sits in the lower
left corner, overlaid on the video: ▲ forward, ▼ reverse, ◀ ▶ spin in
place — ◀ is left wheels backward + right wheels forward (turn left),
▶ is the reverse (turn right). That's how a 4WD car turns, there's
no steering wheel. Hold a button to drive, let go to stop. Drive
speed is fixed at full (100%). On a computer the keyboard arrow keys
drive the same way: hold an arrow to move, let go to stop.
The button in the center of the D-pad (↻ 180°) spins the rover a
half-turn in place: tap it once and it turns around by itself. Any
D-pad button cancels the spin mid-turn. The spin is timed, not
measured — see SPIN_180_TIME to calibrate how far it turns.

LINE FOLLOWING: tap "🤖 Line Follow" and the rover drives itself along
a blue tape line on the floor. The camera looks for the blue line in
the floor just ahead and steers toward it, at full speed (100%). While
line following is on, the video shows a green center box: the rover
drives straight while the line is inside the box. If the line drifts out of the box, the rover stops and
spins in place toward the line until it's back inside, then drives on
(the red dot marks where the camera thinks the line is).
Safety: it stops the moment the line is lost (no creeping forward to
search), if you tap any
D-pad button (manual takes over instantly), or if the page closes or
loses contact for 3 seconds (the page heartbeats while auto is on).
A watchdog also stops the motors within ~1.5 seconds if commands stop
arriving, so the rover can never run away on its own.

SAFETY (BOTH LIDARS): the learned modes (Tour and Go to goal) are
overridden by one rule — if EITHER the LD19's forward arc or the
TF-Luna's low front beam sees an obstacle under the minimum
distance picked on the drive page (0.5 m by default), the rover
stops and turns away instead of following the policy, using the
minimum-angle escape from the full 360° LD19 sweep (left or
right, whichever is the smaller turn), so it gets out of a corner
instead of circling it. It never backs up. If both LiDARs are
offline the rover stops and waits — it never drives blind. The
"Min dist" slider sets that threshold (0.30–2.00 m). Any D-pad
button takes over instantly, and the rover stops if the page
loses contact for 3 seconds.

MANUAL SAFETY: D-pad / keyboard driving is also protected. When
you drive forward or back, the rover slows down as an obstacle
approaches (below 1.0 m) and will not drive into it (stops at
0.30 m). Forward uses the LD19 forward arc + TF-Luna; reverse
uses the LD19 rear arc. The slow-down is enforced both when the
command arrives and continuously between commands (10 Hz), so a
wall that appears suddenly is caught in ~100 ms. Turns (spinning
in place) are not limited. If both LiDARs are offline the rover
drives unprotected, as before — the safety is best-effort. The video shows a cyan marker at
the center where the TF-Luna beam points, labeled with the live
distance ("inf" when nothing is in range). The drive page also
shows a bird's-eye radar view (top right): the rover in the
middle, recent distance readings plotted around it, a dashed red
ring at the picked minimum distance, and ∞ when nothing is in
range.

LD19: the spinning LiDAR (youyeetoo FHL-LD19 kit) is the mapping
sensor. It spins ~10 times a second and measures a full 360-degree
scan, and the drive page's radar draws that scan as cyan dots around
the rover. The TF-Luna's single forward beam still watches the low
front. The LD19's 0-degree direction is the sensor's front; if the
scan looks rotated or mirrored once the sensor is mounted, the
LD19_ANGLE_OFFSET in the code rotates it in software.

GO TO GOAL (LEARNED POLICY): tap the map to set a goal (red marker),
then tap "🤖 Go to goal" and the rover drives itself there using the
policy Walter trained on his Mac (PPO, 500k steps, 87% in sim),
loaded from rover_policy.onnx with ONNX Runtime. It runs at 10 Hz,
reading the live LD19 scan and the map pose. The both-LiDAR safety
rule (above) overrides it when blocked. It stops and says "goal
reached" within 0.25 m of the goal. Any D-pad press takes over
instantly, and it stops if the page loses contact for 3 seconds.
The ONNX file and onnxruntime must be on the Pi (see BUILD_GUIDE.md).

TOUR (THE EXPLORE MODE): tap "🗺️ Tour" and the same policy drives,
but the goal is virtual — every second the rover aims a few metres
out into the most open corridor the LD19 sees, so the goal keeps
moving and the rover tours the house instead of driving to a point.
It never "arrives", so it never stops on its own. This is the
rover's only explore mode now: the old rule-based Obstacle Avoid
was consolidated into it (Nov 24) — the policy already avoids
obstacles, and the rule-based logic lives on as the safety
override above, not as a separate mode. Same D-pad takeover, same
3-second heartbeat. It doesn't remember where it's been (the
policy only sees the LiDAR, not the map), so it can revisit rooms
— a dedicated explore policy would be needed for methodical
coverage.

The camera is shared: one background thread captures frames, and both
the video stream and the line follower read the latest frame, so you
can watch it follow the line live.

The plant monitor normally holds the camera open, so opening the video
feed (or line following) pauses the plant capture service while the
camera is in use, and it resumes when the camera is released. (Needs a
sudoers rule so the server may stop/start plant-capture; see
BUILD_GUIDE.md.)

Motor map (measured Nov 9 with test_motor_map.py — the motors are
wired rear pair / front pair, not left pair / right pair):
    M1 = rear-right   M2 = rear-left   M3 = front-right   M4 = front-left

Same shape as plant-monitor's serve.py (stdlib http.server), on port
8001 so the plant dashboard (8000) keeps running alongside.
"""
import io
import json
import math
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    from adafruit_motorkit import MotorKit
    kit = MotorKit()
    MOTORS_OK = True
except Exception as e:  # library missing or HAT not found — page still loads
    kit = None
    MOTORS_OK = False
    print(f"WARNING: motors unavailable ({e}). Page will load; buttons "
          "will report the problem. Check I2C is on and the HAT is seated.")

try:
    from picamera2 import Picamera2
    PICAM_OK = True
except Exception as e:
    Picamera2 = None
    PICAM_OK = False
    print(f"WARNING: camera unavailable ({e}). Video feed disabled.")

try:
    import cv2
    CV2_OK = True
except Exception:
    cv2 = None
    CV2_OK = False

STATE = {"last_cmd_time": 0.0, "moving": False, "speed": 1.0,
         "throttle_l": 0.0, "throttle_r": 0.0}  # last commanded
# motor throttles (-1..1, negative = forward) — the pose thread
# reads these to dead-reckon where the rover is for the map.
LOCK = threading.Lock()

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1,
      maximum-scale=1, user-scalable=no">
<title>Pi Rover</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden;
         background: #000; }
  body { font-family: -apple-system, sans-serif; color: #eee;
         -webkit-user-select: none; user-select: none;
         touch-action: none; }
  #cam { position: fixed; inset: 0; width: 100%; height: 100%;
         object-fit: contain; background: #000; }
  .overlay { position: fixed; z-index: 10; }
  #title { top: 12px; left: 16px; font-size: 18px; font-weight: 600;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); }
  #camtoggle { top: 10px; right: 12px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #stabtoggle { top: 94px; right: 12px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #stabtoggle.on { background: rgba(56,161,105,0.9); }
  #radartoggle { top: 136px; right: 12px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #radartoggle.on { background: rgba(56,161,105,0.9); }
  #radar { top: 178px; right: 12px; width: 300px; height: 300px;
           border-radius: 12px; background: rgba(0,0,0,0.55);
           pointer-events: none; }
  #autotoggle { top: 52px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #autotoggle.on { background: rgba(56,161,105,0.9); }
  #mindistctl { top: 94px; left: 16px; font-size: 14px; padding: 8px 12px;
           border-radius: 20px; background: rgba(45,55,72,0.8);
           color: #fff; display: flex; gap: 8px; align-items: center; }
  #mindistctl input { width: 88px; accent-color: #48bb78; }
  #mindistval { min-width: 46px; }
  #maptoggle { top: 136px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #maptoggle.on { background: rgba(56,161,105,0.9); }
  #policytoggle { top: 178px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #policytoggle.on { background: rgba(56,161,105,0.9); }
  #tourtoggle { top: 220px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #tourtoggle.on { background: rgba(56,161,105,0.9); }
  #mappanel { top: 60px; left: 50%; transform: translateX(-50%);
           width: min(92vw, 420px); background: rgba(0,0,0,0.85);
           border: 1px solid rgba(255,255,255,0.25); border-radius: 12px;
           padding: 10px; display: none; z-index: 20; }
  #mapimg { width: 100%; aspect-ratio: 1; display: block;
           border-radius: 8px; background: #222; cursor: crosshair; }
  #mapstats { font-size: 13px; color: #a0aec0; margin: 8px 2px 4px; }
  #maphint { font-size: 12px; color: #a0aec0; margin: 0 2px 4px; }
  #mapbtns { display: flex; gap: 8px; }
  #mapbtns button { flex: 1; font-size: 15px; padding: 8px 10px;
           border-radius: 16px; border: none;
           background: rgba(45,55,72,0.9); color: #fff; }
  #camerr { top: 50%; left: 50%; transform: translate(-50%, -50%);
           color: #fc8181; font-size: 18px; text-align: center;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); display: none; }
  #dpad { left: 12px; bottom: 12px; width: 200px; height: 200px;
          border-radius: 50%; background: rgba(255,255,255,0.08);
          border: 2px solid rgba(255,255,255,0.25); }
  #dpad button { position: absolute; width: 64px; height: 64px;
          font-size: 28px; border-radius: 50%;
          border: 2px solid rgba(255,255,255,0.35);
          background: rgba(43,108,176,0.8); color: #fff; }
  #dpad button:active { background: rgba(66,153,225,0.9); }
  #dpad .up { left: 50%; top: 6px; transform: translateX(-50%); }
  #dpad .down { left: 50%; bottom: 6px; transform: translateX(-50%); }
  #dpad .left { left: 6px; top: 50%; transform: translateY(-50%); }
  #dpad .right { right: 6px; top: 50%; transform: translateY(-50%); }
  #dpad .spin { left: 50%; top: 50%; transform: translate(-50%, -50%);
          width: 56px; height: 56px; font-size: 15px; line-height: 1.15; }
  #status { right: 16px; bottom: 12px; font-size: 14px; color: #a0aec0;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); }
</style></head>
<body>
  <img id="cam" src="/video_feed" alt="Rover camera"
       onerror="document.getElementById('camerr').style.display='block'">
  <div id="title" class="overlay">\U0001F916 Pi Rover</div>
  <button id="camtoggle" class="overlay">\U0001F4F7 Camera: On</button>
  <button id="stabtoggle" class="overlay">\U0001F4F9 Stabilize: On</button>
  <button id="radartoggle" class="overlay">\U0001F4E1 Radar: On</button>
  <canvas id="radar" class="overlay" width="300" height="300"></canvas>
  <button id="autotoggle" class="overlay">\U0001F916 Line Follow: Off</button>
  <div id="mindistctl" class="overlay">
    <label for="mindist">Min dist</label>
    <input id="mindist" type="range" min="0.3" max="2" step="0.05" value="0.5">
    <span id="mindistval">0.50 m</span>
  </div>
  <button id="maptoggle" class="overlay">\U0001F5FA Map: Off</button>
  <button id="policytoggle" class="overlay">\U0001F916 Go to goal: Off</button>
  <button id="tourtoggle" class="overlay">\U0001F5FA Tour: Off</button>
  <div id="mappanel" class="overlay">
    <img id="mapimg" src="/map.png" alt="Rover map">
    <div id="mapstats">map loading…</div>
    <div id="maphint">Tap the map to set the goal, then press Go to goal.</div>
    <div id="mapbtns">
      <button id="mapclear">Clear map</button>
      <button id="mapclose">Close</button>
    </div>
  </div>
  <div id="camerr" class="overlay">Camera unavailable &mdash; check the Pi
    log (journalctl -u rover-server).</div>
  <div id="dpad" class="overlay">
    <button class="up" data-cmd="forward">&#9650;</button>
    <button class="down" data-cmd="back">&#9660;</button>
    <button class="left" data-cmd="left">&#9664;</button>
    <button class="right" data-cmd="right">&#9654;</button>
    <button class="spin" id="spinbtn" title="Spin 180°">&#8635;<br>180&deg;</button>
  </div>
  <div id="status" class="overlay"></div>
<script>
  const status = document.getElementById('status');
  let timer = null;
  function send(cmd) {
    fetch('/cmd/' + cmd + '?speed=100', {method: 'POST'})
      .then(r => r.text()).then(t => status.textContent = t)
      .catch(() => status.textContent = 'no connection');
  }
  function start(cmd) {          // hold = send now, then heartbeat
    send(cmd);
    timer = setInterval(() => send(cmd), 500);
  }
  function halt() {
    if (timer) { clearInterval(timer); timer = null; }
    send('stop');
  }
  // Line-follow toggle. While auto is on the page heartbeats every
  // second; the server stops auto if the heartbeat goes quiet.
  const autotoggle = document.getElementById('autotoggle');
  let autoOn = false;
  let autoTimer = null;
  function setAuto(on) {
    autoOn = on;
    autotoggle.textContent = '\U0001F916 Line Follow: ' + (on ? 'On' : 'Off');
    autotoggle.classList.toggle('on', on);
    if (on) {
      fetch('/auto/on', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
      autoTimer = setInterval(() => {
        fetch('/auto/ping', {method: 'POST'})
          .then(r => r.text()).then(t => status.textContent = t)
          .catch(() => { setAuto(false); status.textContent = 'no connection'; });
      }, 1000);
    } else {
      if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
      fetch('/auto/off', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
    }
  }
  autotoggle.addEventListener('click', () => setAuto(!autoOn));
  // Learned policy (ONNX): tap the map to set a goal, then "Go to
  // goal" drives the rover there with the trained model. Same
  // heartbeat pattern as line follow; any D-pad press takes over. If
  // starting fails (no goal set, or the model file is missing) the
  // toggle flips back off and the status says why.
  const policytoggle = document.getElementById('policytoggle');
  let policyOn = false;
  let policyTimer = null;
  function setPolicy(on) {
    policyOn = on;
    policytoggle.textContent = '\U0001F916 Go to goal: ' + (on ? 'On' : 'Off');
    policytoggle.classList.toggle('on', on);
    if (on) {
      if (autoOn) setAuto(false);
      if (tourOn) setTour(false);
      fetch('/policy/on', {method: 'POST'})
        .then(r => r.text().then(t => {
          status.textContent = t;
          if (!r.ok) setPolicy(false);
        }))
        .catch(() => { setPolicy(false); status.textContent = 'no connection'; });
      policyTimer = setInterval(() => {
        fetch('/policy/ping', {method: 'POST'})
          .then(r => r.text()).then(t => status.textContent = t)
          .catch(() => { setPolicy(false); status.textContent = 'no connection'; });
      }, 1000);
    } else {
      if (policyTimer) { clearInterval(policyTimer); policyTimer = null; }
      fetch('/policy/off', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
    }
  }
  policytoggle.addEventListener('click', () => setPolicy(!policyOn));
  // Tour mode: the same learned policy, but the goal is virtual —
  // the rover aims at the most open space and the goal keeps
  // moving, so it tours the house instead of driving to a point.
  // Same heartbeat pattern as Go to goal; any D-pad press takes
  // over. If starting fails (model missing, or the LD19 is
  // offline) the toggle flips back off and the status says why.
  const tourtoggle = document.getElementById('tourtoggle');
  let tourOn = false;
  let tourTimer = null;
  function setTour(on) {
    tourOn = on;
    tourtoggle.textContent = '\U0001F5FA Tour: ' + (on ? 'On' : 'Off');
    tourtoggle.classList.toggle('on', on);
    if (on) {
      if (autoOn) setAuto(false);
      if (policyOn) setPolicy(false);
      fetch('/tour/on', {method: 'POST'})
        .then(r => r.text().then(t => {
          status.textContent = t;
          if (!r.ok) setTour(false);
        }))
        .catch(() => { setTour(false); status.textContent = 'no connection'; });
      tourTimer = setInterval(() => {
        fetch('/tour/ping', {method: 'POST'})
          .then(r => r.text()).then(t => status.textContent = t)
          .catch(() => { setTour(false); status.textContent = 'no connection'; });
      }, 1000);
    } else {
      if (tourTimer) { clearInterval(tourTimer); tourTimer = null; }
      fetch('/tour/off', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
    }
  }
  tourtoggle.addEventListener('click', () => setTour(!tourOn));
  // Minimum safety distance picker: the policy turns away when
  // something gets closer than this (0.30–2.00 m, 0.50 m default).
  // The label follows the slider live; the value is sent when released.
  // Blur after sending so arrow keys drive again right away.
  const mindist = document.getElementById('mindist');
  const mindistval = document.getElementById('mindistval');
  function showMindist() {
    mindistval.textContent = parseFloat(mindist.value).toFixed(2) + ' m';
  }
  mindist.addEventListener('input', showMindist);
  mindist.addEventListener('change', () => {
    showMindist();
    fetch('/safety/mindist?m=' + mindist.value, {method: 'POST'})
      .then(r => r.text()).then(t => status.textContent = t)
      .catch(() => status.textContent = 'no connection');
    mindist.blur();
  });
  showMindist();
  // D-pad buttons: hold to move, let go to stop. Pressing one while
  // auto is on switches back to manual first.
  document.querySelectorAll('button[data-cmd]').forEach(b => {
    const cmd = b.dataset.cmd;
    b.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (autoOn) setAuto(false);
      start(cmd);
    });
    b.addEventListener('pointerup', halt);
    b.addEventListener('pointerleave', halt);
    b.addEventListener('pointercancel', halt);
  });
  // Arrow keys drive too: hold an arrow to move, let go to stop.
  // Same commands and heartbeat as the D-pad; the most recently
  // pressed held key wins, and releasing it falls back to any other
  // arrow still held. Ignored while typing in a field (e.g. the Min
  // dist slider). Losing page focus stops the rover, and the server
  // watchdog is the final backstop if a key-up is ever missed.
  const KEY_CMDS = {ArrowUp: 'forward', ArrowDown: 'back',
                    ArrowLeft: 'left', ArrowRight: 'right'};
  const heldKeys = [];  // arrow keys currently held, most recent last
  let keyCmd = null;
  function keyDrive() {
    const cur = heldKeys.length ?
      KEY_CMDS[heldKeys[heldKeys.length - 1]] : null;
    if (cur === keyCmd) return;
    if (timer) { clearInterval(timer); timer = null; }
    if (cur) {
      if (autoOn) setAuto(false);
      keyCmd = cur;
      start(cur);
    } else {
      keyCmd = null;
      halt();
    }
  }
  document.addEventListener('keydown', e => {
    if (!KEY_CMDS[e.key]) return;
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' ||
              t.tagName === 'SELECT' || t.isContentEditable)) return;
    e.preventDefault();
    if (e.repeat) return;  // heartbeat already running for this key
    if (!heldKeys.includes(e.key)) heldKeys.push(e.key);
    keyDrive();
  });
  document.addEventListener('keyup', e => {
    const i = heldKeys.indexOf(e.key);
    if (i === -1) return;
    heldKeys.splice(i, 1);
    keyDrive();
  });
  window.addEventListener('blur', () => {
    heldKeys.length = 0;
    keyDrive();
  });
  // Center button: spin 180° in place. One tap, the rover turns
  // around and stops by itself. Any D-pad button cancels it mid-spin.
  document.getElementById('spinbtn').addEventListener('click', () => {
    if (autoOn) setAuto(false);
    fetch('/spin180', {method: 'POST'})
      .then(r => r.text()).then(t => status.textContent = t)
      .catch(() => status.textContent = 'no connection');
  });
  // Camera toggle: off frees the camera for the plant monitor.
  const cam = document.getElementById('cam');
  const camtoggle = document.getElementById('camtoggle');
  let camOn = true;
  camtoggle.addEventListener('click', () => {
    camOn = !camOn;
    if (camOn) {
      cam.src = '/video_feed?' + Date.now();
      camtoggle.textContent = '\U0001F4F7 Camera: On';
    } else {
      cam.removeAttribute('src');
      camtoggle.textContent = '\U0001F4F7 Camera: Off';
    }
  });
  // Stabilize toggle: smooths the shaky drive video (on by default).
  const stabtoggle = document.getElementById('stabtoggle');
  let stabOn = true;
  stabtoggle.classList.toggle('on', true);
  stabtoggle.addEventListener('click', () => {
    stabOn = !stabOn;
    stabtoggle.textContent = '\U0001F4F9 Stabilize: ' + (stabOn ? 'On' : 'Off');
    stabtoggle.classList.toggle('on', stabOn);
    fetch('/stabilize/' + (stabOn ? 'on' : 'off'), {method: 'POST'})
      .then(r => r.text()).then(t => status.textContent = t)
      .catch(() => status.textContent = 'no connection');
  });
  // Radar: bird's-eye view from the TF-Luna. The rover sits in the
  // middle pointing up; each dot is a recent distance reading,
  // plotted where it was taken. Dots fade after a few seconds —
  // there is no position tracking, so the picture is only good
  // while the rover turns in place or sits still.
  const radar = document.getElementById('radar');
  const rctx = radar.getContext('2d');
  const radartoggle = document.getElementById('radartoggle');
  let radarOn = true;
  radartoggle.addEventListener('click', () => {
    radarOn = !radarOn;
    radartoggle.textContent = '\U0001F4E1 Radar: ' + (radarOn ? 'On' : 'Off');
    radartoggle.classList.toggle('on', radarOn);
    radar.style.display = radarOn ? 'block' : 'none';
  });
  radartoggle.classList.toggle('on', true);
  const RADAR_MAX_M = 3.0;   // edge of the circle = 3 m
  function drawRadar(data, ld19) {
    const W = radar.width, H = radar.height, cx = W / 2, cy = H / 2;
    const R = 138;           // pixels to the 3 m ring
    rctx.clearRect(0, 0, W, H);
    rctx.strokeStyle = 'rgba(255,255,255,0.25)';
    rctx.lineWidth = 1;
    [1, 2, 3].forEach(m => {
      rctx.beginPath();
      rctx.arc(cx, cy, R * m / RADAR_MAX_M, 0, 2 * Math.PI);
      rctx.stroke();
    });
    rctx.fillStyle = 'rgba(255,255,255,0.5)';
    rctx.font = '13px sans-serif';
    rctx.fillText('1m', cx + 3, cy - R / 3 + 3);
    rctx.fillText('2m', cx + 3, cy - 2 * R / 3 + 3);
    rctx.fillText('3m', cx + 3, cy - R + 10);
    if (data && data.stop_m) {
      // Dashed red ring = the picked minimum safety distance.
      const sr = R * Math.min(data.stop_m, RADAR_MAX_M) / RADAR_MAX_M;
      rctx.strokeStyle = 'rgba(252,129,129,0.8)';
      rctx.setLineDash([4, 4]);
      rctx.beginPath();
      rctx.arc(cx, cy, sr, 0, 2 * Math.PI);
      rctx.stroke();
      rctx.setLineDash([]);
    }
    if (!data || !data.online) {
      rctx.fillStyle = '#fc8181';
      rctx.font = '16px sans-serif';
      rctx.textAlign = 'center';
      rctx.fillText('LiDAR offline', cx, cy + 30);
      rctx.font = '13px sans-serif';
      rctx.fillText((data && data.error) || 'no data', cx, cy + 48);
      rctx.textAlign = 'left';
    } else {
      const heading = data.heading || 0;
      (data.trail || []).forEach(pt => {
        // Heading is a compass bearing (clockwise-positive, 0 =
        // north), the same sense as the radar's clockwise bins,
        // so a point at relative bearing rel plots at +sin / -cos.
        const rel = (pt[0] - heading) * Math.PI / 180;
        const d = Math.min(pt[1], RADAR_MAX_M);
        const r = R * d / RADAR_MAX_M;
        const x = cx + r * Math.sin(rel);
        const y = cy - r * Math.cos(rel);
        const alpha = Math.max(0.15, 1 - pt[2] / 8);
        rctx.fillStyle = 'rgba(252,211,77,' + alpha.toFixed(2) + ')';
        rctx.beginPath();
        rctx.arc(x, y, 4, 0, 2 * Math.PI);
        rctx.fill();
      });
      if (data.dist_m !== null && data.dist_m !== undefined) {
        const d = Math.min(data.dist_m, RADAR_MAX_M);
        const r = R * d / RADAR_MAX_M;
        rctx.fillStyle = '#48bb78';
        rctx.beginPath();
        rctx.arc(cx, cy - r, 7, 0, 2 * Math.PI);
        rctx.fill();
        rctx.fillStyle = '#fff';
        rctx.font = '16px sans-serif';
        rctx.textAlign = 'center';
        rctx.fillText(data.dist_m.toFixed(2) + ' m', cx, cy + 30);
        rctx.textAlign = 'left';
      } else {
        // Sensor alive, nothing in range: infinity, no dot.
        rctx.fillStyle = '#fff';
        rctx.font = '20px sans-serif';
        rctx.textAlign = 'center';
        rctx.fillText('∞', cx, cy + 30);
        rctx.textAlign = 'left';
      }
    }
    if (ld19 && ld19.online && ld19.scan) {
      // LD19 360-degree scan: cyan dots. Angle 0 = forward (up),
      // increasing clockwise, so x = cx + r*sin(a), y = cy - r*cos(a).
      rctx.fillStyle = 'rgba(56,189,248,0.9)';
      ld19.scan.forEach((d, a) => {
        if (d === null || d === undefined) return;
        const dd = Math.min(d, RADAR_MAX_M);
        const r = R * dd / RADAR_MAX_M;
        const rad = a * Math.PI / 180;
        const x = cx + r * Math.sin(rad);
        const y = cy - r * Math.cos(rad);
        rctx.beginPath();
        rctx.arc(x, y, 3, 0, 2 * Math.PI);
        rctx.fill();
      });
      rctx.fillStyle = 'rgba(255,255,255,0.6)';
      rctx.font = '13px sans-serif';
      rctx.textAlign = 'center';
      rctx.fillText('LD19 ' + (ld19.hz || 0).toFixed(1) + ' Hz', cx, cy + 62);
      rctx.textAlign = 'left';
    }
    rctx.fillStyle = '#4299e1';
    rctx.beginPath();
    rctx.moveTo(cx, cy - 15);
    rctx.lineTo(cx - 10, cy + 12);
    rctx.lineTo(cx + 10, cy + 12);
    rctx.closePath();
    rctx.fill();
  }
  function refreshRadar() {
    if (!radarOn) return;
    Promise.all([
      fetch('/lidar').then(r => r.json()).catch(() => null),
      fetch('/ld19').then(r => r.json()).catch(() => null)
    ]).then(([lidar, ld19]) => drawRadar(lidar, ld19));
  }
  setInterval(refreshRadar, 250);
  refreshRadar();

  // Map: floor plan built from the LD19. The panel sits over the
  // video (the D-pad stays visible, so you can drive while the map
  // is open). It refreshes once a second while open; mapping keeps
  // running in the background when it is closed.
  const mappanel = document.getElementById('mappanel');
  const mapimg = document.getElementById('mapimg');
  const mapstats = document.getElementById('mapstats');
  const maptoggle = document.getElementById('maptoggle');
  let mapOn = false;
  let mapTimer = null;
  function refreshMap() {
    mapimg.src = '/map.png?t=' + Date.now();
    fetch('/map.json').then(r => r.json()).then(s => {
      let txt = 'x ' + s.x + ' m, y ' + s.y + ' m, heading ' +
        s.heading + '° — ' + s.scans + ' scans, ' + s.occupied +
        ' wall cells, ' + s.free + ' free cells';
      if (s.goal_x !== null && s.goal_x !== undefined)
        txt += ' — goal (' + s.goal_x + ', ' + s.goal_y + ')';
      mapstats.textContent = txt;
    }).catch(() => { mapstats.textContent = 'map stats unavailable'; });
  }
  function setMap(on) {
    mapOn = on;
    maptoggle.textContent = '\U0001F5FA Map: ' + (on ? 'On' : 'Off');
    maptoggle.classList.toggle('on', on);
    mappanel.style.display = on ? 'block' : 'none';
    if (on) { refreshMap(); mapTimer = setInterval(refreshMap, 1000); }
    else if (mapTimer) { clearInterval(mapTimer); mapTimer = null; }
  }
  maptoggle.addEventListener('click', () => setMap(!mapOn));
  document.getElementById('mapclose').addEventListener('click', () => setMap(false));
  document.getElementById('mapclear').addEventListener('click', () => {
    fetch('/map/clear', {method: 'POST'}).then(r => r.text())
      .then(t => { status.textContent = t; refreshMap(); });
  });
  // Tap the map to set the goal. The map image is 400x400 px (1 px
  // = 1 cell = 5 cm), centered on (0, 0) at pixel (200, 200), so a
  // tap at pixel (px, py) is x = (px-200)*0.05, y = (200-py)*0.05.
  // Works whether the policy is on or off — when on, the rover
  // retargets live.
  mapimg.addEventListener('click', (e) => {
    const rect = mapimg.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width * 400;
    const py = (e.clientY - rect.top) / rect.height * 400;
    const gx = (px - 200) * 0.05;
    const gy = (200 - py) * 0.05;
    fetch('/policy/goal?x=' + gx.toFixed(2) + '&y=' + gy.toFixed(2),
      {method: 'POST'})
      .then(r => r.text()).then(t => {
        status.textContent = t;
        refreshMap();
      })
      .catch(() => status.textContent = 'no connection');
  });
</script>
</body></html>
"""


def set_motors(left, right):
    """left/right: -1.0..1.0 applied to both motors on that side.

    Motor map, MEASURED Nov 9 with test_motor_map.py (the motors are
    wired rear pair / front pair, not left pair / right pair):
        M1 = rear-right   M2 = rear-left
        M3 = front-right  M4 = front-left
    """
    if kit is None:
        return
    kit.motor2.throttle = left    # rear-left
    kit.motor4.throttle = left    # front-left
    kit.motor1.throttle = right   # rear-right
    kit.motor3.throttle = right   # front-right


def stop_all():
    if kit is None:
        return
    for m in (kit.motor1, kit.motor2, kit.motor3, kit.motor4):
        m.throttle = None


def apply_command(cmd, speed_pct):
    s = max(20, min(100, speed_pct)) / 100.0
    with LOCK:
        STATE["last_cmd_time"] = time.time()
        STATE["speed"] = s
        if cmd == "forward":  # reversed Nov 8: rover drove backward
            set_motors(-s, -s)
            STATE["throttle_l"] = -s
            STATE["throttle_r"] = -s
        elif cmd == "back":
            set_motors(s, s)
            STATE["throttle_l"] = s
            STATE["throttle_r"] = s
        elif cmd == "left":      # spin in place: left wheels back,
            set_motors(s, -s)    # right wheels forward = turn left
            STATE["throttle_l"] = s
            STATE["throttle_r"] = -s
        elif cmd == "right":     # spin in place: left wheels forward,
            set_motors(-s, s)    # right wheels back = turn right
            STATE["throttle_l"] = -s
            STATE["throttle_r"] = s
        elif cmd == "stop":
            stop_all()
            STATE["throttle_l"] = 0.0
            STATE["throttle_r"] = 0.0
        else:
            return False
        STATE["moving"] = cmd != "stop"
    return True


def apply_drive(left, right):
    """Direct left/right throttles in -1.0..1.0 (used by line following
    and the /drive test endpoint). Forward is negative throttles."""
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))
    with LOCK:
        STATE["last_cmd_time"] = time.time()
        STATE["throttle_l"] = left
        STATE["throttle_r"] = right
        if left == 0 and right == 0:
            stop_all()
            STATE["moving"] = False
        else:
            set_motors(left, right)
            STATE["moving"] = True
    return True


def watchdog():
    # The phone page heartbeats while a button is held / auto is on.
    # No heartbeat for 1.5s = let go, phone asleep, or Wi-Fi lost -> stop.
    while True:
        time.sleep(0.25)
        with LOCK:
            if STATE["moving"] and time.time() - STATE["last_cmd_time"] > 1.5:
                stop_all()
                STATE["moving"] = False


def plant_capture_active():
    r = subprocess.run(["systemctl", "is-active", "--quiet", "plant-capture"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def set_plant_capture(want_on):
    action = "start" if want_on else "stop"
    subprocess.run(["sudo", "systemctl", action, "plant-capture"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def encode_jpeg(frame_bgr):
    """frame is a BGR numpy array. Picamera2 delivers RGB888 captures in
    BGR channel order, which is exactly what OpenCV expects, so the frame
    goes straight into imencode with NO conversion. Converting it again
    (RGB->BGR) swaps red and blue a second time and turns everything blue.
    """
    if CV2_OK:
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg.tobytes() if ok else None
    try:
        import numpy as np
        from PIL import Image
        buf = io.BytesIO()
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return None


def annotate_frame(frame):
    """Draw overlays onto a copy of the frame for the video stream:
    the green center target box (only while line following is on),
    the last detected line position as a red dot, and the cyan
    LiDAR beam marker at the center with the live distance.
    Detection itself always uses the raw frame."""
    if not CV2_OK:
        return frame
    h, w = frame.shape[:2]
    disp = frame.copy()
    y0, y1 = int(h * 0.55), int(h * 0.95)
    if AUTO["active"]:
        # Center target box: only while line following is on.
        x0 = int(w * (0.5 - CENTER_BOX_HALF))
        x1 = int(w * (0.5 + CENTER_BOX_HALF))
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
    if LAST_DETECT["found"]:
        cx = int(w / 2 + LAST_DETECT["offset"] * (w / 2))
        cv2.circle(disp, (cx, (y0 + y1) // 2), 10, (0, 0, 255), -1)
    # LiDAR beam marker: the sensor looks straight ahead, so the
    # beam lands at the center of the picture (roughly — it sits
    # a little high when the camera tilts down a lot). The label
    # is the live distance, or "inf" when nothing is in range
    # (the video text font has no infinity glyph; the radar shows
    # the real ∞ sign). Nothing drawn while the sensor is offline.
    reading = lidar_reading()
    if reading is not None:
        dist_m, _strength = reading
        cx, cy = w // 2, h // 2
        cv2.circle(disp, (cx, cy), 20, (255, 255, 0), 2)
        cv2.line(disp, (cx - 30, cy), (cx - 20, cy), (255, 255, 0), 2)
        cv2.line(disp, (cx + 20, cy), (cx + 30, cy), (255, 255, 0), 2)
        cv2.line(disp, (cx, cy - 30), (cx, cy - 20), (255, 255, 0), 2)
        cv2.line(disp, (cx, cy + 20), (cx, cy + 30), (255, 255, 0), 2)
        label = "inf" if dist_m is None else f"{dist_m:.2f} m"
        cv2.putText(disp, label, (cx - 32, cy + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(disp, label, (cx - 32, cy + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return disp


class VideoStabilizer:
    """Digital video stabilization for the shaky drive video: the
    motors vibrate the rigid camera mount, so the picture jitters
    frame to frame. Estimates the frame-to-frame shift with phase
    correlation, accumulates a camera trajectory, fits a straight
    line to the recent trajectory (the intended motion — driving
    forward, turning), and warps each frame by the deviation from
    that line (the shake). A straight-line fit has no lag on steady
    motion, so driving forward doesn't shift the picture. Only the
    video stream is stabilized; the detectors keep the raw frames."""
    def __init__(self, window=15):
        self.window = window
        self.prev_gray = None
        self.traj = []
        self.t = 0

    def reset(self):
        self.prev_gray = None
        self.traj = []
        self.t = 0

    def stabilize(self, frame):
        import numpy as np
        if not CV2_OK:
            return frame
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                           (320, 180)).astype(np.float32)  # phaseCorrelate needs float
        if self.prev_gray is None:
            self.prev_gray = small
            self.traj = [(0, 0.0, 0.0)]
            self.t = 0
            return frame
        shift, response = cv2.phaseCorrelate(self.prev_gray, small)
        dx, dy = float(shift[0]), float(shift[1])
        self.prev_gray = small
        # Unreliable match (featureless view) or a huge shift (scene
        # cut, fast spin) — re-anchor instead of warping wildly.
        if response < 0.2 or abs(dx) > 25 or abs(dy) > 25:
            self.traj = [(self.t, 0.0, 0.0)]
            return frame
        self.t += 1
        px, py = self.traj[-1][1], self.traj[-1][2]
        nx, ny = px + dx, py + dy
        self.traj.append((self.t, nx, ny))
        if len(self.traj) > self.window:
            self.traj.pop(0)
        if len(self.traj) < 3:
            return frame
        ts = np.array([p[0] for p in self.traj], dtype=np.float64)
        xs = np.array([p[1] for p in self.traj], dtype=np.float64)
        ys = np.array([p[2] for p in self.traj], dtype=np.float64)
        fx = np.polyfit(ts, xs, 1)
        fy = np.polyfit(ts, ys, 1)
        # Deviation of the current position from the fitted line
        # (the shake), scaled up from the 320x180 measurement (x4).
        dev_x = (fx[0] * self.t + fx[1]) - nx
        dev_y = (fy[0] * self.t + fy[1]) - ny
        M = np.float32([[1, 0, dev_x * 4], [0, 1, dev_y * 4]])
        h, w = frame.shape[:2]
        return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)


STABILIZER = VideoStabilizer()
STABILIZE = {"on": True}  # video stabilization, toggled from the page


class CameraManager:
    """One shared camera: a background thread captures frames; the MJPEG
    stream and the line follower both read the latest frame, so you can
    watch the rover follow the line live."""

    def __init__(self):
        self.cond = threading.Condition()
        self.latest_jpg = None
        self.latest_frame = None
        self.seq = 0
        self.users = 0
        self.running = False
        self.picam = None
        self.thread = None
        self.was_plant_running = False

    def acquire(self):
        """Take a share of the camera; starts it on first use.
        Returns True if the camera is running."""
        with self.cond:
            self.users += 1
            if not self.running:
                try:
                    self._start_locked()
                except Exception as e:
                    print(f"camera failed to start: {e}", flush=True)
                    self.users -= 1
                    return False
            return True

    def release(self):
        with self.cond:
            self.users = max(0, self.users - 1)
            if self.users == 0 and self.running:
                self._stop_locked()

    def _start_locked(self):
        if not PICAM_OK:
            raise RuntimeError("camera unavailable on this Pi")
        # The plant monitor holds the camera open; pause it while in use.
        self.was_plant_running = plant_capture_active()
        if self.was_plant_running:
            print("video: pausing plant-capture while streaming", flush=True)
            set_plant_capture(False)
        picam = Picamera2()
        # Widest view: the IMX708 sensor is natively 16:9, so a 16:9
        # stream (1280x720) uses the full sensor width.
        picam.configure(picam.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}))
        try:
            from libcamera import controls
            picam.set_controls({
                # Fixed focus: continuous autofocus hunts every time the
                # rover moves and is slow to re-lock, so the video goes
                # soft while driving. Lock it at ~0.7 m (LensPosition
                # is in dioptres: 1 / distance in metres) — that covers
                # the floor tape just ahead and obstacles a metre or
                # two out. If the picture is soft, tune this number:
                # bigger = focuses closer, smaller = focuses farther.
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": 1.4,
                # Cap the frame time at 1/60s: at 1/30s any movement
                # smears. Shorter exposure = sharper while driving;
                # indoors it costs some brightness / more gain noise.
                "FrameDurationLimits": (10000, 16666),
            })
        except Exception:
            pass
        picam.start()
        time.sleep(0.5)  # warmup / exposure settle
        self.picam = picam
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _stop_locked(self):
        self.running = False
        thread, picam = self.thread, self.picam
        self.thread = None
        self.picam = None
        if thread is not None:
            thread.join(timeout=2)
        if picam is not None:
            try:
                picam.stop()
            except Exception:
                pass
            try:
                picam.close()
            except Exception:
                pass
        if self.was_plant_running:
            print("video: resuming plant-capture", flush=True)
            set_plant_capture(True)
            self.was_plant_running = False

    def _capture_loop(self):
        while self.running:
            try:
                frame = self.picam.capture_array()
            except Exception as e:
                print(f"camera capture failed: {e}", flush=True)
                break
            jpg = encode_jpeg(annotate_frame(
                STABILIZER.stabilize(frame) if STABILIZE["on"] else frame))
            with self.cond:
                self.latest_frame = frame
                if jpg is not None:
                    self.latest_jpg = jpg
                self.seq += 1
                self.cond.notify_all()
            time.sleep(0.03)  # ~25 fps capture

    def get_frame(self):
        with self.cond:
            return self.latest_frame

    def wait_jpg(self, last_seq, timeout=2.0):
        """Wait for a frame newer than last_seq; returns (jpg, seq)."""
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout=timeout)
            return self.latest_jpg, self.seq


CAM = CameraManager()


def detect_line(frame):
    """Find the blue tape line in a BGR frame. Looks at the band of floor
    just ahead of the rover (lower part of the image) for blue pixels.
    Returns (found, offset, blue_px): offset in [-1, 1], + means the line
    is to the right of center. Uses HSV hue when OpenCV is available
    (robust to warm indoor light, which shifts blue tape toward yellow);
    falls back to a blue-dominant BGR test."""
    import numpy as np
    h, w = frame.shape[:2]
    band = frame[int(h * 0.55):int(h * 0.95)]
    if CV2_OK:
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 70, 50]),
                           np.array([135, 255, 255]))
        count = int((mask > 0).sum())
    else:
        b16 = band.astype(np.int16)
        b, g, r = b16[..., 0], b16[..., 1], b16[..., 2]
        mask = (b > 80) & (b - r > 40) & (b >= g)  # blue tape, not the floor
        count = int(mask.sum())
    if count < 500:  # too few blue pixels: no line in view
        return False, 0.0, count
    _, xs = np.where(mask > 0)
    cx = float(xs.mean())
    offset = (cx - w / 2) / (w / 2)
    return True, max(-1.0, min(1.0, offset)), count


def debug_stats():
    """Snapshot of what the line detector sees, for the /debug page."""
    import numpy as np
    frame = CAM.get_frame()
    if frame is None:
        return None
    h, w = frame.shape[:2]
    band = frame[int(h * 0.55):int(h * 0.95)]
    b, g, r = (float(band[..., i].mean()) for i in (0, 1, 2))
    b16 = band.astype(np.int16)
    bgr_count = int(((b16[..., 0] > 80) & (b16[..., 0] - b16[..., 2] > 40)
                     & (b16[..., 0] >= b16[..., 1])).sum())
    whole = frame.astype(np.int16)
    whole_count = int(((whole[..., 0] > 80) & (whole[..., 0] - whole[..., 2] > 40)
                       & (whole[..., 0] >= whole[..., 1])).sum())
    hsv_count = None
    if CV2_OK:
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        hsv_count = int((cv2.inRange(hsv, np.array([90, 70, 50]),
                                     np.array([135, 255, 255])) > 0).sum())
    found, offset, count = detect_line(frame)
    with AUTO_LOCK:
        status = AUTO["status"]
    return "\n".join([
        f"frame: {w}x{h}",
        f"band rows: {int(h * 0.55)}-{int(h * 0.95)}",
        f"band mean BGR: {b:.0f} / {g:.0f} / {r:.0f}",
        f"blue px in band (HSV): {hsv_count}",
        f"blue px in band (BGR test): {bgr_count}",
        f"blue px in whole frame (BGR test): {whole_count}",
        f"detector: found={found} offset={offset:+.2f} count={count}",
        f"auto status: {status}",
        f"lidar: online={LIDAR['online']} dist={LIDAR['dist_m']} "
        f"strength={LIDAR['strength']} frames={LIDAR['frames']} "
        f"bad={LIDAR['bad']} err={LIDAR['error'] or 'none'}",
        f"ld19: online={LD19['online']} port={LD19['port']} "
        f"hz={LD19['hz']:.1f} points={LD19['points']} "
        f"packets={LD19['packets']} bad={LD19['bad']} "
        f"err={LD19['error'] or 'none'}",
        f"heading: {HEADING['deg']:.0f} deg trail: {len(TRAIL)} pts",
        f"map: {map_stats()}",
    ])


AUTO = {"active": False, "cam": False, "last_ping": 0.0,
        "status": "auto off"}
AUTO_LOCK = threading.Lock()
# Line-follow speeds: low power can't move the loaded rover, so
# everything runs at 100% — cruise and spin.
AUTO_BASE_SPEED = 1.0
AUTO_SPIN_SPEED = 1.0  # spin-in-place speed when adjusting angle
# Center box: the line counts as centered while |offset| <= 0.3, i.e.
# its centroid sits in the middle 30% of the image width (35-65%).
# The video shows this box in green; the red dot marks the last
# detected line position.
CENTER_BOX_HALF = 0.3
LAST_DETECT = {"found": False, "offset": 0.0}
# 180° spin: how long to spin in place at full power. There is no gyro
# or encoder feedback, so the turn is timed, not measured — calibrate
# it by testing: if the rover turns less than a half-turn, raise this
# number; if it overshoots past facing backward, lower it.
SPIN_180_TIME = 1.5  # seconds at 100% power
# (Was 2.0 while the motor map was wrong and the spin barely moved;
# with the map fixed the spin is much faster, so 2.0 overshot. 1.0
# undershot — 1.5 is the next calibration try. Overshoot -> lower,
# undershoot -> raise.)
SPIN = {"active": False, "cancel": False}
SPIN_LOCK = threading.Lock()


def auto_set_status(text):
    with AUTO_LOCK:
        AUTO["status"] = text


def auto_stop(reason):
    """Stop auto mode (and the motors) and give the camera back."""
    with AUTO_LOCK:
        was_active = AUTO["active"]
        AUTO["active"] = False
        AUTO["status"] = reason
        cam = AUTO["cam"]
        AUTO["cam"] = False
    apply_drive(0, 0)
    if cam:
        CAM.release()
    print(f"auto: {reason}", flush=True)
    return was_active


def auto_loop():
    """Line-following control loop: ~10 Hz, proportional steering."""
    while True:
        with AUTO_LOCK:
            active = AUTO["active"]
            last_ping = AUTO["last_ping"]
        if not active:
            time.sleep(0.1)
            continue
        # The page heartbeats while auto is on; silence = page gone.
        if time.time() - last_ping > 3.0:
            auto_stop("page gone — auto stopped")
            time.sleep(0.1)
            continue
        frame = CAM.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        found, offset, count = detect_line(frame)
        LAST_DETECT["found"] = found
        LAST_DETECT["offset"] = offset
        if not found:
            # Line lost: stop right here. No creeping forward to search —
            # the rover would drive off the line blind.
            auto_stop("line lost — stopped")
            continue
        elif abs(offset) <= CENTER_BOX_HALF:
            # Line is inside the center box: drive straight.
            apply_drive(-AUTO_BASE_SPEED, -AUTO_BASE_SPEED)
            auto_set_status("following • in box")
        else:
            # Line is outside the box: STOP forward motion and spin in
            # place toward the line until it comes back into the box.
            # Uses the same left/right mapping as the D-pad buttons
            # (confirmed correct), so the turn goes the right way.
            side = "left" if offset < 0 else "right"
            apply_command(side, int(AUTO_SPIN_SPEED * 100))
            auto_set_status(f"adjusting • turning {side}")
        time.sleep(0.1)


# --- TF-Luna LiDAR -------------------------------------------------------
# Single-point time-of-flight LiDAR on the USB-TTL adapter (see
# BUILD_GUIDE.md for wiring). It streams 9-byte frames at 100 Hz:
# 0x59 0x59, distance cm (little-endian), strength, chip temp,
# checksum. The reader thread keeps the latest reading in LIDAR and
# a short trail of recent readings (heading + distance) for the
# bird's-eye radar view. If pyserial is missing or the adapter is
# unplugged, the reader just reports offline — the server and the
# camera-based fallback keep working.
LIDAR_PORT = "/dev/ttyUSB0"
LIDAR_BAUD = 115200
LIDAR = {"online": False, "dist_m": None, "strength": None,
         "temp_c": None, "updated": 0.0, "frames": 0, "bad": 0,
         "error": ""}
LIDAR_LOCK = threading.Lock()
# Heading estimate for the radar trail: degrees, clockwise
# positive like a compass bearing (0 = north, 90 = east; a right
# turn adds, a left turn subtracts). The pose thread owns it (dead
# reckoning from the commanded motor throttles) — there is no gyro
# or encoder, so it drifts. The trail fades in seconds, so a rough
# heading is good enough for the picture.
HEADING = {"deg": 0.0}
TRAIL = []  # (heading_deg, dist_m, timestamp), pruned to the last 8 s
TRAIL_MAX_AGE = 8.0
# Turn rate at 100% power, from the SPIN_180_TIME calibration
# (180° in SPIN_180_TIME seconds).
TURN_RATE_DEG_S = 180.0 / SPIN_180_TIME


def lidar_extract(buf):
    """Pull complete TF-Luna frames out of buf (a bytearray).
    Returns a list of (dist_m, strength, temp_c). Drops garbage
    and bad-checksum frames; a partial trailing frame stays in
    buf for the next read."""
    readings = []
    while True:
        i = buf.find(b"\x59\x59")
        if i < 0:
            # No header: keep only a possible split-header tail byte.
            del buf[:-1]
            break
        if len(buf) - i < 9:
            # Partial frame: drop garbage before the header, keep
            # the partial frame for the next read.
            del buf[:i]
            break
        frame = bytes(buf[i:i + 9])
        del buf[:i + 9]
        if (sum(frame[:8]) & 0xFF) != frame[8]:
            with LIDAR_LOCK:
                LIDAR["bad"] += 1
            continue
        dist_cm = frame[2] | (frame[3] << 8)
        strength = frame[4] | (frame[5] << 8)
        temp_raw = frame[6] | (frame[7] << 8)
        readings.append((dist_cm / 100.0, strength,
                         (temp_raw / 8.0) - 256.0))
    return readings


def lidar_loop():
    """Read the TF-Luna forever, updating LIDAR + TRAIL. Never
    raises: a missing adapter or bad port just means offline."""
    try:
        import serial
    except ImportError:
        with LIDAR_LOCK:
            LIDAR["error"] = ("pyserial not installed "
                              "(sudo apt install python3-serial)")
        return
    buf = bytearray()
    last_sample = 0.0
    while True:
        try:
            ser = serial.Serial(LIDAR_PORT, LIDAR_BAUD, timeout=0.2)
        except Exception as e:
            with LIDAR_LOCK:
                LIDAR["online"] = False
                LIDAR["error"] = f"cannot open {LIDAR_PORT}: {e}"
            time.sleep(5.0)
            continue
        with LIDAR_LOCK:
            LIDAR["error"] = ""
        try:
            while True:
                chunk = ser.read(ser.in_waiting or 1)
                if chunk:
                    buf += chunk
                    for dist_m, strength, temp_c in lidar_extract(buf):
                        now = time.time()
                        # A 0.00 m reading with a weak return means
                        # nothing in range (no obstacle). A 0 with a
                        # strong return means something too close to
                        # measure — keep that as a real reading.
                        # No-obstacle is stored as None so displays
                        # show infinity instead of a bogus 0.00 m.
                        no_obstacle = (dist_m == 0.0 and strength < 100)
                        with LIDAR_LOCK:
                            LIDAR["online"] = True
                            LIDAR["dist_m"] = None if no_obstacle else dist_m
                            LIDAR["strength"] = strength
                            LIDAR["temp_c"] = temp_c
                            LIDAR["updated"] = now
                            LIDAR["frames"] += 1
                            if now - last_sample >= 0.2:
                                last_sample = now
                                if not no_obstacle:
                                    TRAIL.append((HEADING["deg"], dist_m, now))
                                cutoff = now - TRAIL_MAX_AGE
                                while TRAIL and TRAIL[0][2] < cutoff:
                                    TRAIL.pop(0)
                else:
                    time.sleep(0.01)
        except Exception as e:
            with LIDAR_LOCK:
                LIDAR["online"] = False
                LIDAR["error"] = f"read error: {e}"
            try:
                ser.close()
            except Exception:
                pass
            buf.clear()
            time.sleep(2.0)


def lidar_reading():
    """Latest LiDAR reading as (dist_m, strength), or None if the
    sensor is offline or the reading is stale (>1 s old). dist_m is
    None when the sensor is alive but sees no obstacle in range
    (display as infinity, and treat as clear)."""
    with LIDAR_LOCK:
        if not LIDAR["online"]:
            return None
        if time.time() - LIDAR["updated"] > 1.0:
            return None
        return LIDAR["dist_m"], LIDAR["strength"]


# ------------------------------------------------------------------ LD19
# LD19 spinning LiDAR (youyeetoo FHL-LD19 kit, 12 m) — the mapping
# sensor. It streams 47-byte packets at 230400 baud, ~10 Hz spin,
# 360-degree scan, one-way (nothing to send it). This thread keeps
# the latest distance per degree in LD19["bins"] (a live 360-degree
# picture, refreshed ~10x a second) and the /ld19 endpoint serves it
# to the drive page, which draws the scan as cyan dots on the radar.
# The safety override uses both LiDARs: the LD19's forward arc watches
# the surroundings in front (see ld19_forward_min) and the TF-Luna
# watches the low front — if either reports an obstacle under the
# minimum distance, the rover stops and turns away (minimum-angle
# escape).
# The reader auto-detects its port by data signature (0x54 header +
# CRC8), skipping the TF-Luna's port, so the two adapters are told
# apart by what they send, not by name. If the LD19 is unplugged or
# pyserial is missing, this just reports offline — the server, the
# TF-Luna safety check, and the camera keep working.
# Angle: left-handed, sensor front = 0 deg, increasing clockwise.
# LD19_ANGLE_OFFSET rotates the scan if the sensor's 0 deg does not
# point at the rover's front once mounted — calibrate by pointing
# the rover at a wall and checking the front sector reads it.
# Calibrated Nov 23, 2026: the sensor's 0 deg points at the rover's
# left, so with no offset the rover's right side showed at the front
# (top) of the radar. -90 deg was tried first and rotated it the wrong
# way (front ended up at the back, right on the left). +90 deg is the
# correct rotation: front reads at the top, right on the right.
LD19_BAUD = 230400
LD19_MAX_M = 12.0
LD19_ANGLE_OFFSET = 90.0
LD19_AVOID_HALF_ANGLE = 30  # avoid watches the LD19's forward arc,
# +/-30 deg around bin 0 (rover front) — wide enough to catch an
# obstacle the narrow TF-Luna beam threads past, narrow enough not
# to treat a side wall as a head-on block.
MANUAL_SLOW_M = 1.0  # manual driving: below this distance the
# rover slows down as it approaches an obstacle.
MANUAL_STOP_M = 0.30  # manual driving: at or below this distance
# the rover will not drive toward the obstacle at all.
MANUAL_REAR_HALF_ANGLE = 45  # reverse protection watches the
# LD19's rear arc, +/-45 deg around bin 180 (rover back).
LD19 = {"online": False, "port": None, "hz": 0.0, "points": 0,
        "packets": 0, "bad": 0, "updated": 0.0, "error": "",
        "bins": [None] * 360}  # whole degree -> latest dist (m) or None
LD19_LOCK = threading.Lock()


def _ld19_crc_table():
    """CRC-8, poly 0x4D, init 0x00, MSB-first (LDROBOT LD19 manual)."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_LD19_CRC_TABLE = _ld19_crc_table()


def ld19_crc8(data):
    crc = 0
    for b in data:
        crc = _LD19_CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


def ld19_extract(buf):
    """Pull complete LD19 packets out of buf (a bytearray).
    Returns a list of (points, speed_dps); points = [(angle_deg,
    dist_m, intensity)]. Drops garbage and bad-CRC packets; a
    partial trailing packet stays in buf for the next read."""
    out = []
    while True:
        i = buf.find(b"\x54")
        if i < 0:
            del buf[:]
            break
        if len(buf) - i < 47:
            del buf[:i]
            break
        pkt = bytes(buf[i:i + 47])
        del buf[:i + 47]
        if pkt[1] != 0x2C:
            with LD19_LOCK:
                LD19["bad"] += 1
            continue
        if ld19_crc8(pkt[:-1]) != pkt[-1]:
            with LD19_LOCK:
                LD19["bad"] += 1
            continue
        speed = pkt[2] | (pkt[3] << 8)  # deg/s
        start = (pkt[4] | (pkt[5] << 8)) / 100.0
        end = (pkt[42] | (pkt[43] << 8)) / 100.0
        if end < start:
            end += 360.0
        step = (end - start) / 11.0
        pts = []
        for k in range(12):
            off = 6 + k * 3
            dist_mm = pkt[off] | (pkt[off + 1] << 8)
            inten = pkt[off + 2]
            ang = (start + step * k) % 360.0
            pts.append((ang, dist_mm / 1000.0, inten))
        out.append((pts, speed))
    return out


def ld19_find_port():
    """Find the LD19's serial port by its data signature. Opens each
    /dev/ttyUSB* port except the TF-Luna's at 230400 baud and looks
    for valid LD19 packets (0x54 header + CRC). Returns the port, or
    None if no LD19 stream is found."""
    import glob
    try:
        import serial
    except ImportError:
        return None
    for port in sorted(glob.glob("/dev/ttyUSB*")):
        if port == LIDAR_PORT:
            continue  # the TF-Luna lives here — never steal its port
        try:
            ser = serial.Serial(port, LD19_BAUD, timeout=0.2)
        except Exception:
            continue
        try:
            buf = bytearray()
            good = 0
            t_end = time.time() + 1.5
            while time.time() < t_end and good < 3:
                chunk = ser.read(4096)
                if chunk:
                    buf.extend(chunk)
                    good += len(ld19_extract(buf))
                    if good >= 3:
                        break
            if good >= 3:
                return port
        except Exception:
            pass
        finally:
            try:
                ser.close()
            except Exception:
                pass
    return None


def ld19_loop():
    """Read the LD19 forever, updating LD19["bins"]. Never raises:
    a missing adapter or bad port just means offline."""
    try:
        import serial
    except ImportError:
        with LD19_LOCK:
            LD19["error"] = ("pyserial not installed "
                             "(sudo apt install python3-serial)")
        return
    buf = bytearray()
    while True:
        port = ld19_find_port()
        if port is None:
            with LD19_LOCK:
                LD19["online"] = False
                LD19["error"] = "no LD19 on any /dev/ttyUSB* port"
            time.sleep(5.0)
            continue
        with LD19_LOCK:
            LD19["port"] = port
            LD19["error"] = ""
        try:
            ser = serial.Serial(port, LD19_BAUD, timeout=0.2)
        except Exception as e:
            with LD19_LOCK:
                LD19["online"] = False
                LD19["error"] = f"cannot open {port}: {e}"
            time.sleep(5.0)
            continue
        try:
            while True:
                chunk = ser.read(ser.in_waiting or 1)
                if chunk:
                    buf += chunk
                    for pts, speed in ld19_extract(buf):
                        now = time.time()
                        with LD19_LOCK:
                            LD19["online"] = True
                            LD19["updated"] = now
                            LD19["packets"] += 1
                            hz = speed / 360.0
                            LD19["hz"] = (hz if LD19["hz"] == 0.0
                                          else 0.8 * LD19["hz"] + 0.2 * hz)
                            for ang, dist_m, _inten in pts:
                                idx = int((ang + LD19_ANGLE_OFFSET) % 360.0)
                                old = LD19["bins"][idx]
                                if 0.0 < dist_m <= LD19_MAX_M:
                                    LD19["bins"][idx] = dist_m
                                    if old is None:
                                        LD19["points"] += 1
                                elif dist_m == 0.0:
                                    # No return in this direction:
                                    # clear the bin so the radar
                                    # doesn't show a stale point.
                                    LD19["bins"][idx] = None
                                    if old is not None:
                                        LD19["points"] -= 1
                else:
                    time.sleep(0.01)
        except Exception as e:
            with LD19_LOCK:
                LD19["online"] = False
                LD19["error"] = f"read error: {e}"
            try:
                ser.close()
            except Exception:
                pass
            buf.clear()
            time.sleep(2.0)


def ld19_forward_min():
    """Closest LD19 return in the forward arc, as (dist_m, angle_deg),
    or None if the LD19 is offline or stale (>1 s). Returns
    (inf, None) when it is alive but sees nothing in the arc (clear).
    angle_deg is signed: negative = obstacle on the rover's left,
    positive = on the right (bin 0 = front, clockwise positive)."""
    with LD19_LOCK:
        if not LD19["online"]:
            return None
        if time.time() - LD19["updated"] > 1.0:
            return None
        bins = list(LD19["bins"])
    best = None
    best_ang = None
    for a in range(-LD19_AVOID_HALF_ANGLE, LD19_AVOID_HALF_ANGLE + 1):
        d = bins[a % 360]
        if d is None:
            continue
        if best is None or d < best:
            best = d
            best_ang = a
    if best is None:
        return (float("inf"), None)
    return (best, best_ang)


def ld19_rear_min():
    """Closest LD19 return in the rear arc, as (dist_m, angle_deg),
    or None if the LD19 is offline or stale (>1 s). Returns
    (inf, None) when it is alive but sees nothing in the arc
    (clear). Used for reverse protection in manual driving."""
    with LD19_LOCK:
        if not LD19["online"]:
            return None
        if time.time() - LD19["updated"] > 1.0:
            return None
        bins = list(LD19["bins"])
    best = None
    best_ang = None
    for a in range(180 - MANUAL_REAR_HALF_ANGLE,
                   180 + MANUAL_REAR_HALF_ANGLE + 1):
        d = bins[a % 360]
        if d is None:
            continue
        if best is None or d < best:
            best = d
            best_ang = a
    if best is None:
        return (float("inf"), None)
    return (best, best_ang)


def _manual_obstacle_dist(cmd):
    """Closest obstacle distance (m) in the direction a manual
    command would move the rover, or None if neither sensor has
    data for that direction. Forward uses the LD19 forward arc
    and the TF-Luna (closest of the two); reverse uses the LD19
    rear arc. Returns inf when the sensors are alive and clear."""
    if cmd == "forward":
        dists = []
        fwd = ld19_forward_min()
        if fwd is not None:
            dists.append(fwd[0])
        tf = lidar_reading()
        if tf is not None and tf[0] is not None and tf[0] > 0:
            dists.append(tf[0])
        if not dists:
            return None
        return min(dists)
    if cmd == "back":
        rear = ld19_rear_min()
        if rear is None:
            return None
        return rear[0]
    return None


def manual_speed_factor(cmd):
    """Speed factor [0.0, 1.0] for a manual drive command, from the
    obstacle distance in that direction. 1.0 = full speed (clear,
    or no sensor data — the rover drives unprotected, as before);
    0.0 = stop (at or inside MANUAL_STOP_M); linear between
    MANUAL_STOP_M and MANUAL_SLOW_M. Turns and stop are unscaled."""
    if cmd not in ("forward", "back"):
        return 1.0
    dist = _manual_obstacle_dist(cmd)
    if dist is None or dist == float("inf"):
        return 1.0
    if dist <= MANUAL_STOP_M:
        return 0.0
    if dist >= MANUAL_SLOW_M:
        return 1.0
    return (dist - MANUAL_STOP_M) / (MANUAL_SLOW_M - MANUAL_STOP_M)


def manual_safety_loop():
    """Continuously caps manual driving speed between /cmd/ repeats
    (the page re-sends a held button every 500 ms; this loop runs
    at 10 Hz so a wall that appears suddenly is caught in ~100 ms).
    Only ever slows the rover down, never speeds it up — the next
    /cmd/ re-send restores speed when the path clears. Does nothing
    when auto / tour / policy is running (those have their own
    safety override) or when the rover is not moving manually."""
    while True:
        time.sleep(0.1)
        with AUTO_LOCK:
            auto_on = AUTO["active"]
        with POLICY_LOCK:
            policy_on = POLICY["active"]
        with TOUR_LOCK:
            tour_on = TOUR["active"]
        if auto_on or policy_on or tour_on:
            continue
        with LOCK:
            if not STATE["moving"]:
                continue
            tl, tr = STATE["throttle_l"], STATE["throttle_r"]
            if time.time() - STATE["last_cmd_time"] > 1.5:
                continue
        # Both throttles negative = forward; both positive = back;
        # mixed = spinning in place (no translation, no cap needed).
        if tl < 0 and tr < 0:
            cmd = "forward"
        elif tl > 0 and tr > 0:
            cmd = "back"
        else:
            continue
        factor = manual_speed_factor(cmd)
        allowed = factor  # full speed is throttle 1.0
        # The motors need ~20% to move the loaded rover (same
        # floor as apply_command): below that the rover stalls, so
        # clamp the crawl to 20% and stop only when the factor
        # itself is 0 — otherwise the rover creeps forever at a
        # throttle too small to matter.
        if 0 < allowed < 0.20:
            allowed = 0.20
        mag = max(abs(tl), abs(tr))
        if mag > allowed:
            with LOCK:
                # Re-check under the lock: only cap if still moving
                # the same way (a new command may have arrived).
                if STATE["moving"] and STATE["throttle_l"] == tl \
                        and STATE["throttle_r"] == tr:
                    if allowed <= 0:
                        stop_all()
                        STATE["throttle_l"] = 0.0
                        STATE["throttle_r"] = 0.0
                        STATE["moving"] = False
                    else:
                        nl = -allowed if tl < 0 else allowed
                        nr = -allowed if tr < 0 else allowed
                        set_motors(nl, nr)
                        STATE["throttle_l"] = nl
                        STATE["throttle_r"] = nr


ESCAPE_MARGIN_M = 0.2  # an escape arc must be clear past
# stop_m + this, so the rover has room to drive forward after it
# turns instead of getting blocked again at once.

def ld19_escape_direction(stop_m):
    """Minimum-angle escape from the full 360-degree LD19 scan.
    Finds the nearest direction the rover can face such that its
    forward arc (±LD19_AVOID_HALF_ANGLE) is clear past
    stop_m + ESCAPE_MARGIN_M. Returns ("left", angle_deg) or
    ("right", angle_deg) — the smaller turn that gets the front
    pointing at open space — or None if the LD19 is offline/stale
    or no direction is clear. Bins are clockwise from the front,
    so a target at bin b needs a right turn of b degrees (b <= 180)
    or a left turn of 360 - b (b > 180). The search starts at 1
    degree (never 0) so a TF-Luna-only block still turns away from
    the low obstacle instead of reporting 'already clear'."""
    with LD19_LOCK:
        if not LD19["online"]:
            return None
        if time.time() - LD19["updated"] > 1.0:
            return None
        bins = list(LD19["bins"])
    clear_m = stop_m + ESCAPE_MARGIN_M
    half = LD19_AVOID_HALF_ANGLE

    def arc_clear(center):
        """True if every bin in ±half around center is clear (no
        return, or farther than clear_m)."""
        for a in range(center - half, center + half + 1):
            d = bins[a % 360]
            if d is not None and d <= clear_m:
                return False
        return True

    # Search outward from the front; at each turn angle check the
    # right side (bin t) and the left side (bin 360 - t). Return
    # the first clear direction. On a tie (both sides clear at the
    # same angle) prefer the side whose arc has the larger minimum
    # distance — the more open side.
    for t in range(1, 181):
        right_ok = arc_clear(t)
        left_ok = arc_clear(360 - t)
        if right_ok and left_ok:
            def arc_min(center):
                ds = [bins[a % 360]
                      for a in range(center - half, center + half + 1)]
                ds = [d for d in ds if d is not None]
                return min(ds) if ds else float("inf")
            if arc_min(t) >= arc_min(360 - t):
                return ("right", t)
            return ("left", t)
        if right_ok:
            return ("right", t)
        if left_ok:
            return ("left", t)
    return None


# Safety distance. The minimum distance (m) under which the
# both-LiDAR safety override stops the policy and turns away
# (minimum-angle escape from the full LD19 sweep — left or right,
# whichever is the smaller turn; left only as a fallback when the
# LD19 has no answer). The override reads the forward distance
# from both LiDARs: the LD19's forward arc watches the
# surroundings, the TF-Luna watches the low front. It never backs
# up.
LIDAR_STOP_M = 0.5     # default minimum safety distance (m)
# The minimum safety distance is pickable on the drive page
# (0.30–2.00 m); the picked value lives here and resets to the
# default when the server restarts.
MINDIST_MIN_M = 0.3
MINDIST_MAX_M = 2.0
MINDIST = {"m": LIDAR_STOP_M}
MINDIST_LOCK = threading.Lock()


def get_mindist():
    with MINDIST_LOCK:
        return MINDIST["m"]


def set_mindist(m):
    """Clamp a picked minimum safety distance and store it."""
    m = max(MINDIST_MIN_M, min(MINDIST_MAX_M, float(m)))
    with MINDIST_LOCK:
        MINDIST["m"] = m
    return m


# --- Pose (dead reckoning) + occupancy-grid map -------------------------
# The map is a floor plan built from the LD19's 360-degree scans.
# The rover has no wheel encoders or gyro, so its pose (x, y,
# heading) starts as dead reckoning from the commanded motor
# throttles — but every scan is then matched against the map so
# far (scan_match), which corrects the pose, so the forward-speed
# guess and the dead-reckoned position don't matter: the LiDAR
# data pins the pose. The map thread reads the latest LD19 scan +
# pose ~2x a second, corrects the pose, and ray-casts each return
# into the grid: cells along the ray are free, the hit cell is
# occupied. The drive page shows it as a Map panel (/map.png).
FORWARD_SPEED_M_S = 0.3  # m/s at full throttle — CALIBRATE: drive a
# known straight distance and compare it to the map; if the map
# shows the rover going farther than it really did, lower this.
MAP_SIZE_M = 20.0   # the map covers a 20 x 20 m square, centered on
# the rover's starting spot. Drive beyond the edge and the map just
# clips (points outside are dropped).
MAP_CELL_M = 0.05   # 5 cm per cell
MAP_CELLS = int(MAP_SIZE_M / MAP_CELL_M)  # 400 x 400
MAP_CENTER = MAP_CELLS // 2  # grid cell of the rover's start (0, 0)

# Pose: metres, metres, degrees (0 = up/north, clockwise positive).
POSE = {"x": 0.0, "y": 0.0, "heading": 0.0}
POSE_LOCK = threading.Lock()
# Grid: 0 = unknown, 1 = free, 2 = occupied (numpy uint8, made lazily).
MAP = {"grid": None, "scans": 0, "updated": 0.0}
MAP_LOCK = threading.RLock()  # reentrant: scan_match reads the grid
# while map_update_once (same thread) writes it.


def map_ensure_grid():
    """Create the occupancy grid on first use (numpy). Returns True
    if the grid exists, False if numpy is missing (map disabled)."""
    if MAP["grid"] is not None:
        return True
    try:
        import numpy as np
    except ImportError:
        return False
    MAP["grid"] = np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.uint8)
    return True


def pose_loop():
    """Dead-reckoning pose, ~10 Hz. Reads the last commanded motor
    throttles and integrates the pose: forward speed from the mean
    throttle, turn rate from the throttle difference. This is the
    SOLE owner of the heading — spin_180 used to credit its turn
    via heading_add, but that double-counted once the pose thread
    integrated the same turn from the throttles, so that call is
    gone. HEADING mirrors POSE for the radar trail
    and the /lidar endpoint."""
    last = time.time()
    while True:
        time.sleep(0.1)
        now = time.time()
        dt = now - last
        last = now
        if dt <= 0 or dt > 1.0:
            continue
        with LOCK:
            moving = STATE["moving"]
            last_cmd = STATE["last_cmd_time"]
            tl = STATE["throttle_l"]
            tr = STATE["throttle_r"]
        if not moving or now - last_cmd > 1.5:
            continue
        # Wheel speeds (m/s, positive = forward). Forward is
        # negative throttle, so flip the sign.
        v = -FORWARD_SPEED_M_S * (tl + tr) / 2.0
        # Turn rate (deg/s). Heading is a compass bearing:
        # clockwise-positive, 0 = north, 90 = east. A left turn
        # (tl=+1, tr=-1, left wheels back / right wheels forward)
        # rotates the rover counterclockwise, so it DECREASES the
        # bearing — hence the minus sign. (This sign was flipped
        # until Nov 23: the heading grew on left turns, which
        # mirrored the pose and smeared the map into a blob while
        # the radar — which doesn't use the pose — still looked
        # fine.)
        w_deg = -TURN_RATE_DEG_S * (tl - tr) / 2.0
        with POSE_LOCK:
            POSE["heading"] = (POSE["heading"] + w_deg * dt) % 360.0
            hr = math.radians(POSE["heading"])
            POSE["x"] += v * math.sin(hr) * dt
            POSE["y"] += v * math.cos(hr) * dt
            heading = POSE["heading"]
        with LIDAR_LOCK:
            HEADING["deg"] = heading


def scan_match(bins, px, py, ph):
    """Correct the pose (px, py, ph) by matching the scan (bins) to
    the map — this is what makes the forward-speed guess and the
    dead-reckoned position not matter. Uses a coarse-to-fine grid
    search around the estimate: it scores every candidate pose by
    how well the scan aligns with the map (+1 per scan point that
    lands on a mapped wall, -1 per point on mapped free space,
    0 on unknown) and keeps the best. When the rover is exploring
    new ground the score is flat and the pose stays at the estimate
    (nothing to match against yet); when it revisits mapped walls
    the match pulls the pose back. Returns (cx, cy, ch)."""
    if not map_ensure_grid():
        return (px, py, ph)
    import numpy as np
    # Scan points in the rover's frame (metres, +y = rover forward,
    # +x = rover right). Bins with no return are skipped.
    pts = []
    for b, dist in enumerate(bins):
        if dist is None or dist <= 0 or dist > LD19_MAX_M:
            continue
        a = math.radians(b)
        pts.append((dist * math.sin(a), dist * math.cos(a)))
    if len(pts) < 10:
        return (px, py, ph)  # too few points to match
    pts = np.array(pts)  # (N, 2), rover frame
    grid = MAP["grid"]

    def score(cx, cy, ch):
        """+1 per point on a mapped wall, -1 per point on mapped
        free space, 0 on unknown. Higher = better aligned."""
        hr = math.radians(ch)
        s, c = math.sin(hr), math.cos(hr)
        # Rover frame -> world: forward is (sin, cos), right is
        # (cos, -sin) in (x, y).
        wx = cx + pts[:, 0] * c + pts[:, 1] * s
        wy = cy - pts[:, 0] * s + pts[:, 1] * c
        cols = MAP_CENTER + np.round(wx / MAP_CELL_M).astype(int)
        rows = MAP_CENTER - np.round(wy / MAP_CELL_M).astype(int)
        ok = ((cols >= 0) & (cols < MAP_CELLS)
              & (rows >= 0) & (rows < MAP_CELLS))
        if not ok.any():
            return 0.0
        vals = grid[rows[ok], cols[ok]]
        return float(((vals == 2).sum() - (vals == 1).sum()))

    with MAP_LOCK:
        # Coarse-to-fine grid search around the estimate. A random
        # hill climb gets stuck in local maxima on this spiky
        # exact-cell score (the true pose scores ~354/360 while a
        # nearby wrong pose scores ~140), so instead we exhaustively
        # check a grid of candidates and keep the best — the true
        # pose is the clear global maximum.
        def search(cx, cy, ch, dxy, dth):
            best = (cx, cy, ch)
            best_s = score(cx, cy, ch)
            for ddx in (-dxy, -dxy / 2, 0.0, dxy / 2, dxy):
                for ddy in (-dxy, -dxy / 2, 0.0, dxy / 2, dxy):
                    for ddt in (-dth, -dth / 2, 0.0, dth / 2, dth):
                        cand = (cx + ddx, cy + ddy,
                                (ch + ddt) % 360.0)
                        s = score(*cand)
                        if s > best_s:
                            best_s = s
                            best = cand
            return best, best_s

        # Coarse: +-0.3 m, +-12 deg. Fine: +-0.06 m, +-3 deg.
        best, _ = search(px, py, ph, 0.30, 12.0)
        best, _ = search(best[0], best[1], best[2], 0.06, 3.0)
        return best


def map_update_once(bins, px, py, ph):
    """One map update from a scan (bins = 360 distances or None)
    at pose (px, py, ph). Ray-casts each return into the grid:
    cells along the ray are free (only if still unknown, so a stray
    clear reading can't erase a wall), the hit cell is occupied.
    Bins with no return mean clear space to the LD19's 12 m max.
    Returns (occupied, free) cell counts, or None if the grid is
    unavailable or the rover is off the map."""
    if not map_ensure_grid():
        return None
    import numpy as np
    rc = MAP_CENTER + int(round(px / MAP_CELL_M))
    rr = MAP_CENTER - int(round(py / MAP_CELL_M))
    if not (0 <= rc < MAP_CELLS and 0 <= rr < MAP_CELLS):
        return None
    grid = MAP["grid"]
    with MAP_LOCK:
        for b, dist in enumerate(bins):
            # World angle of this bin (deg, clockwise from up).
            wa = math.radians((ph + b) % 360.0)
            dx = math.sin(wa)
            dy = math.cos(wa)
            if dist is None:
                rng = LD19_MAX_M  # no return: clear to max range
                hit = False
            else:
                rng = dist
                hit = True
            ex = px + rng * dx
            ey = py + rng * dy
            ec = MAP_CENTER + int(round(ex / MAP_CELL_M))
            er = MAP_CENTER - int(round(ey / MAP_CELL_M))
            steps = max(abs(ec - rc), abs(er - rr), 1)
            xs = np.linspace(rc, ec, steps + 1).round().astype(int)
            ys = np.linspace(rr, er, steps + 1).round().astype(int)
            ok = ((xs >= 0) & (xs < MAP_CELLS)
                  & (ys >= 0) & (ys < MAP_CELLS))
            xs, ys = xs[ok], ys[ok]
            if len(xs) == 0:
                continue
            # Free along the ray (not the hit cell itself, and
            # never overwrite an occupied cell).
            if len(xs) > 1:
                fx, fy = xs[:-1], ys[:-1]
                free_mask = grid[fy, fx] == 0
                grid[fy[free_mask], fx[free_mask]] = 1
            # Occupied at the hit (not the rover's own cell).
            if hit and (xs[-1] != rc or ys[-1] != rr):
                grid[ys[-1], xs[-1]] = 2
        MAP["scans"] += 1
        MAP["updated"] = time.time()
        return (int((grid == 2).sum()), int((grid == 1).sum()))


def map_update_loop():
    """Build the occupancy grid from the LD19, ~2 Hz. Each cycle it
    takes the latest scan + dead-reckoned pose, corrects the pose
    by scan-matching the scan against the map so far (so the
    forward-speed guess and the dead-reckoned position don't
    matter — the LiDAR data pins the pose), writes the corrected
    pose back to POSE, and feeds the scan to map_update_once at
    the corrected pose. Points outside the 20 m square are
    dropped."""
    while True:
        time.sleep(0.5)
        now = time.time()
        with LD19_LOCK:
            if not (LD19["online"] and now - LD19["updated"] < 1.0):
                continue
            bins = list(LD19["bins"])
        with POSE_LOCK:
            px, py, ph = POSE["x"], POSE["y"], POSE["heading"]
        # Scan-match: correct the dead-reckoned pose against the
        # map, then write the correction back so the pose thread
        # keeps integrating from the corrected pose.
        cx, cy, ch = scan_match(bins, px, py, ph)
        with POSE_LOCK:
            POSE["x"], POSE["y"], POSE["heading"] = cx, cy, ch
        map_update_once(bins, cx, cy, ch)


def map_render_png():
    """Render the occupancy grid + rover to a PNG (bytes), or None
    if the map/cv2 isn't available. Unknown = dark gray, free =
    white, occupied = black, rover = green triangle."""
    if not CV2_OK or not map_ensure_grid():
        return None
    import numpy as np
    with MAP_LOCK:
        grid = MAP["grid"].copy()
    with POSE_LOCK:
        px, py, ph = POSE["x"], POSE["y"], POSE["heading"]
    img = np.full(grid.shape, 60, dtype=np.uint8)
    img[grid == 1] = 240
    img[grid == 2] = 0
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rc = MAP_CENTER + int(round(px / MAP_CELL_M))
    rr = MAP_CENTER - int(round(py / MAP_CELL_M))
    if 0 <= rc < MAP_CELLS and 0 <= rr < MAP_CELLS:
        hr = math.radians(ph)
        s, c = math.sin(hr), math.cos(hr)
        nose = (int(rc + 10 * s), int(rr - 10 * c))
        b1 = (int(rc - 6 * s + 6 * c), int(rr + 6 * c + 6 * s))
        b2 = (int(rc - 6 * s - 6 * c), int(rr + 6 * c - 6 * s))
        cv2.fillPoly(bgr, [np.array([nose, b1, b2])], (0, 200, 0))
    # Goal marker (red) — the policy's target, set by tapping the map.
    with POLICY_LOCK:
        gx = POLICY["goal_x"]
        gy = POLICY["goal_y"]
    if gx is not None and gy is not None:
        gc = MAP_CENTER + int(round(gx / MAP_CELL_M))
        gr = MAP_CENTER - int(round(gy / MAP_CELL_M))
        if 0 <= gc < MAP_CELLS and 0 <= gr < MAP_CELLS:
            cv2.circle(bgr, (gc, gr), 6, (0, 0, 255), 2)
            cv2.circle(bgr, (gc, gr), 2, (0, 0, 255), -1)
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else None


def map_clear():
    """Reset the grid to unknown and the pose to the start (0, 0,
    heading 0)."""
    if map_ensure_grid():
        with MAP_LOCK:
            MAP["grid"][:] = 0
            MAP["scans"] = 0
            MAP["updated"] = 0.0
    with POSE_LOCK:
        POSE["x"] = 0.0
        POSE["y"] = 0.0
        POSE["heading"] = 0.0
    with LIDAR_LOCK:
        HEADING["deg"] = 0.0
        TRAIL.clear()


def map_stats():
    """Pose + grid counts, for /map.json and /debug."""
    if not map_ensure_grid():
        return None
    import numpy as np
    with MAP_LOCK:
        grid = MAP["grid"]
        occ = int((grid == 2).sum())
        free = int((grid == 1).sum())
        scans = MAP["scans"]
    with POSE_LOCK:
        x, y, h = POSE["x"], POSE["y"], POSE["heading"]
    with POLICY_LOCK:
        gx = POLICY["goal_x"]
        gy = POLICY["goal_y"]
    return {"x": round(x, 2), "y": round(y, 2),
            "heading": round(h, 1), "occupied": occ, "free": free,
            "scans": scans, "size_m": MAP_SIZE_M, "cell_m": MAP_CELL_M,
            "goal_x": (round(gx, 2) if gx is not None else None),
            "goal_y": (round(gy, 2) if gy is not None else None)}


# --- Learned policy (ONNX) — go to a goal --------------------------------
# The go-to-goal policy Walter trained on his Mac (PPO, 500k steps,
# 87% success in sim) runs on the Pi as an ONNX file. The Pi never
# trains; it loads rover_policy.onnx with ONNX Runtime and runs it
# at 10 Hz — the same step rate the policy was trained at.
#
# The observation is built exactly the way RoverEnv builds it:
# 24 LD19 sectors (min distance per 15-degree sector, /12 m) +
# goal bearing (sin, cos, relative to the heading) + goal distance
# (/5 m). The LD19 bins are already indexed by angle from the
# rover's front (bin 0 = front, clockwise), which is the same frame
# the training rays use, so sector i is bins [i*15, i*15+15).
# The model returns one int action: 0 = forward, 1 = left, 2 =
# right, 3 = stop — the same commands the drive page sends.
#
# Safety is the both-LiDAR rule, and it
# overrides the policy: if EITHER the LD19 forward arc or the
# TF-Luna reports an obstacle under the picked minimum distance,
# the rover turns (minimum-angle escape) instead of following the
# policy; if both LiDARs are offline it stops and waits. The
# policy was trained to avoid obstacles itself — the override is
# a backstop for the sim-to-real gap, not the navigator.
# The goal is set by tapping the map on the drive page; the rover
# stops and reports "goal reached" within 0.25 m of it.
POLICY_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rover_policy.onnx")
POLICY_GOAL_REACHED_M = 0.25  # GOAL_R in rover_env.py
POLICY_DT = 0.1  # 10 Hz, the training step rate
POLICY = {"active": False, "last_ping": 0.0, "status": "policy off",
          "goal_x": None, "goal_y": None, "model": None,
          "model_error": "", "tour": False, "goal_virtual": False}
POLICY_LOCK = threading.Lock()

# --- Tour mode: the same policy, no fixed goal ---------------------------
# Tour uses the go-to-goal policy with a virtual goal that points
# at open space: every second (or when the rover gets close to the
# current virtual goal) it picks the most open corridor from the
# LD19 scan and aims a few metres out into it. The goal keeps
# moving as the rover moves, so it never "arrives" — it just keeps
# following open space through the house. No retraining, no new
# model file; the tour is a goal source, the policy does the
# driving. The same both-LiDAR safety override applies.
TOUR_REPICK_S = 1.0  # pick a new virtual goal this often
TOUR_GOAL_DIST_M = 2.5  # virtual goal this far out (m)
TOUR_GOAL_CLOSE_M = 0.8  # repick when this close to the virtual goal
TOUR = {"active": False, "last_ping": 0.0, "status": "tour off",
        "goal_x": None, "goal_y": None, "repick_at": 0.0}
TOUR_LOCK = threading.Lock()


def policy_load_model():
    """Load rover_policy.onnx with ONNX Runtime (lazy, once).
    Returns True if the model is ready. On failure sets
    POLICY['model_error'] with the reason (missing file or missing
    onnxruntime) and returns False — the page reports it."""
    with POLICY_LOCK:
        if POLICY["model"] is not None:
            return True
        if not os.path.exists(POLICY_ONNX_PATH):
            POLICY["model_error"] = (
                "rover_policy.onnx not found — copy it to "
                "~/pi-monitoring-rover/ (scp from the Mac)")
            return False
        try:
            import onnxruntime as ort
        except ImportError:
            POLICY["model_error"] = (
                "onnxruntime not installed — run: pip install "
                "--break-system-packages onnxruntime")
            return False
        try:
            POLICY["model"] = ort.InferenceSession(
                POLICY_ONNX_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            POLICY["model_error"] = f"could not load model: {e}"
            return False
        POLICY["model_error"] = ""
        return True


def policy_observation():
    """Build the 27-float observation the policy was trained on,
    from the live LD19 scan + pose + goal. Returns None if the
    LD19 is offline/stale or no goal is set. Mirrors RoverEnv._obs
    exactly (see rl_rover/rover_env.py)."""
    with POLICY_LOCK:
        gx = POLICY["goal_x"]
        gy = POLICY["goal_y"]
    if gx is None or gy is None:
        return None
    with LD19_LOCK:
        if not LD19["online"] or time.time() - LD19["updated"] > 1.0:
            return None
        bins = list(LD19["bins"])
    with POSE_LOCK:
        px, py, ph = POSE["x"], POSE["y"], POSE["heading"]
    # 24 sectors: min distance per 15-degree sector; a missing
    # return counts as max range (12 m), same as the training sim.
    import numpy as np
    sectors = []
    for i in range(24):
        ds = [bins[(i * 15 + k) % 360] for k in range(15)]
        ds = [d for d in ds if d is not None]
        sectors.append((min(ds) if ds else 12.0) / 12.0)
    # Bearing to the goal, relative to the heading (compass bearing,
    # 0 = north/+y, clockwise-positive — atan2(dx, dy) is the
    # bearing from +y, and the relative bearing wraps to [-180, 180]).
    bearing = math.degrees(math.atan2(gx - px, gy - py))
    rel = math.radians((bearing - ph + 180.0) % 360.0 - 180.0)
    gd = math.hypot(gx - px, gy - py) / 5.0
    obs = np.array(sectors + [math.sin(rel), math.cos(rel),
                              min(gd, 1.0)], dtype=np.float32)
    return obs


def policy_goal_dist():
    """Distance (m) from the rover's pose to the goal, or None."""
    with POLICY_LOCK:
        gx = POLICY["goal_x"]
        gy = POLICY["goal_y"]
    if gx is None or gy is None:
        return None
    with POSE_LOCK:
        px, py = POSE["x"], POSE["y"]
    return math.hypot(gx - px, gy - py)


def policy_set_status(text):
    with POLICY_LOCK:
        POLICY["status"] = text


def policy_stop(reason):
    """Stop policy mode (and the motors)."""
    with POLICY_LOCK:
        was_active = POLICY["active"]
        POLICY["active"] = False
        POLICY["status"] = reason
    if was_active:
        apply_drive(0, 0)


def tour_goal_direction():
    """Pick the tour direction from the full LD19 scan: the
    direction whose ±20° arc has the largest minimum distance —
    the widest open corridor. Returns (angle_deg, min_dist_m),
    angle clockwise from the rover's front, or None if the LD19
    is offline/stale."""
    with LD19_LOCK:
        if not LD19["online"] or time.time() - LD19["updated"] > 1.0:
            return None
        bins = list(LD19["bins"])
    best_ang = None
    best_min = -1.0
    half = 20
    for c in range(0, 360, 5):
        ds = [bins[(c + k) % 360] for k in range(-half, half + 1)]
        ds = [d for d in ds if d is not None]
        m = min(ds) if ds else 12.0
        if m > best_min:
            best_min = m
            best_ang = c
    if best_ang is None:
        return None
    return (best_ang, best_min)


def tour_pick_goal():
    """Set the policy goal to a virtual tour target: a point out
    in the most open direction from the LD19 scan. Returns True
    if a goal was set, False if the LD19 has no answer."""
    res = tour_goal_direction()
    if res is None:
        return False
    ang_deg, min_dist = res
    # Goal this far out, inside the clear corridor (not at the
    # wall); never closer than 0.5 m so it stays ahead of the rover.
    gd = min(min_dist * 0.8, TOUR_GOAL_DIST_M)
    if gd < 0.5:
        gd = 0.5
    with POSE_LOCK:
        px, py, ph = POSE["x"], POSE["y"], POSE["heading"]
    # ang is clockwise from the rover's front; the front points
    # along the heading, so the absolute bearing is heading + ang.
    bearing = math.radians((ph + ang_deg) % 360.0)
    gx = px + gd * math.sin(bearing)
    gy = py + gd * math.cos(bearing)
    with POLICY_LOCK:
        POLICY["goal_x"] = gx
        POLICY["goal_y"] = gy
        POLICY["goal_virtual"] = True
    with TOUR_LOCK:
        TOUR["goal_x"] = gx
        TOUR["goal_y"] = gy
    return True


def tour_set_status(text):
    with TOUR_LOCK:
        TOUR["status"] = text


def tour_stop(reason):
    """Stop tour mode (and the policy + motors). Clears the policy
    goal only if it's a virtual tour goal — a fixed goal the user
    tapped stays, so switching from tour to "Go to goal" keeps it."""
    with TOUR_LOCK:
        was_active = TOUR["active"]
        TOUR["active"] = False
        TOUR["status"] = reason
        TOUR["goal_x"] = None
        TOUR["goal_y"] = None
    with POLICY_LOCK:
        POLICY["tour"] = False
        if POLICY["goal_virtual"]:
            POLICY["goal_x"] = None
            POLICY["goal_y"] = None
            POLICY["goal_virtual"] = False
    policy_stop(reason)


def policy_loop():
    """Run the trained policy at 10 Hz while active. Each step:
    build the observation, run the ONNX model, drive the action.
    The both-LiDAR safety check runs first and overrides the
    policy when blocked (minimum-angle escape turn); both LiDARs
    offline means stop and wait. Reaching the
    goal (<= 0.25 m) stops the rover and ends the run — except in
    tour mode, where the goal is virtual and keeps moving: reaching
    it just picks a new one, so the rover never stops. Stops if
    the page heartbeat is gone for 3 s."""
    blocked_since = None
    escape_dir = "left"
    escape_angle = 0
    while True:
        with POLICY_LOCK:
            active = POLICY["active"]
            last_ping = POLICY["last_ping"]
            tour_on = POLICY["tour"]
        if not active:
            time.sleep(0.1)
            continue
        if time.time() - last_ping > 3.0:
            if tour_on:
                tour_stop("page gone — tour stopped")
            else:
                policy_stop("page gone — policy stopped")
            time.sleep(0.1)
            continue
        # Tour mode: the goal is virtual and moves. Repick it every
        # second, or when the rover gets close to it, so it stays
        # ahead and the rover never "arrives".
        if tour_on:
            now = time.time()
            with TOUR_LOCK:
                repick_at = TOUR["repick_at"]
            gd = policy_goal_dist()
            if now >= repick_at or (gd is not None
                                    and gd < TOUR_GOAL_CLOSE_M):
                if tour_pick_goal():
                    with TOUR_LOCK:
                        TOUR["repick_at"] = now + TOUR_REPICK_S
        # Goal reached?
        gd = policy_goal_dist()
        if gd is not None and gd <= POLICY_GOAL_REACHED_M:
            if tour_on:
                # Virtual goal reached — pick a new one, keep going.
                tour_pick_goal()
                with TOUR_LOCK:
                    TOUR["repick_at"] = time.time() + TOUR_REPICK_S
                time.sleep(0.1)
                continue
            policy_stop("goal reached 🎯")
            time.sleep(0.1)
            continue
        # Need a fresh LD19 scan for the observation; never drive
        # blind on a stale scan.
        obs = policy_observation()
        if obs is None:
            apply_drive(0, 0)
            policy_set_status("LD19 offline — stopped")
            time.sleep(0.1)
            continue
        # --- safety override (both-LiDAR rule) ---
        ld = ld19_forward_min()
        tf = lidar_reading()
        if ld is None and tf is None:
            apply_drive(0, 0)
            policy_set_status("both lidars offline — stopped")
            time.sleep(0.1)
            continue
        stop_m = get_mindist()
        ld_m = None
        if ld is not None and ld[0] != float("inf"):
            ld_m = ld[0]
        tf_m = None
        tf_bad = False
        if tf is not None:
            tf_m, tf_strength = tf
            if tf_m is not None and not (
                    0.0 <= tf_m <= 8.0 and 0 < tf_strength < 65535):
                tf_bad = True
                tf_m = None
        if tf_bad and ld is None:
            apply_drive(0, 0)
            policy_set_status("lidar bad reading — stopped")
            time.sleep(0.1)
            continue
        cands = []
        if ld_m is not None:
            cands.append(ld_m)
        if tf_m is not None:
            cands.append(tf_m)
        blocked = bool(cands) and min(cands) < stop_m
        if blocked:
            if blocked_since is None:
                blocked_since = time.time()
                esc = ld19_escape_direction(stop_m)
                if esc is not None:
                    escape_dir, escape_angle = esc
                else:
                    escape_dir, escape_angle = "left", 0
            policy_set_status(
                f"policy blocked — turning {escape_dir} {escape_angle}°")
            apply_command(escape_dir, 100)
            time.sleep(POLICY_DT)
            continue
        blocked_since = None
        escape_dir = "left"
        escape_angle = 0
        # --- policy action ---
        with POLICY_LOCK:
            sess = POLICY["model"]
        if sess is None:
            apply_drive(0, 0)
            policy_set_status("model not loaded — stopped")
            time.sleep(0.1)
            continue
        try:
            action = int(sess.run(None, {"observation": obs[None, :]})[0][0])
        except Exception as e:
            apply_drive(0, 0)
            policy_set_status(f"model error — stopped ({e})")
            time.sleep(0.1)
            continue
        cmds = ("forward", "left", "right", "stop")
        cmd = cmds[action] if 0 <= action <= 3 else "stop"
        if gd is not None:
            policy_set_status(f"policy: {cmd} — goal {gd:.2f} m away")
        else:
            policy_set_status(f"policy: {cmd}")
        apply_command(cmd, 100)
        time.sleep(POLICY_DT)


def spin_180():
    """Spin in place for SPIN_180_TIME at full power, then stop.
    Runs in its own thread; any D-pad press cancels it mid-turn.
    Direction doesn't matter for a half-turn — it ends facing back
    either way — so it just uses the same spin as the right button."""
    with SPIN_LOCK:
        if SPIN["active"]:
            return False
        SPIN["active"] = True
        SPIN["cancel"] = False
    # A manual move: stop auto / tour / policy first, like a D-pad
    # press does.
    with AUTO_LOCK:
        auto_was_on = AUTO["active"]
    if auto_was_on:
        auto_stop("spin 180")
    with TOUR_LOCK:
        tour_was_on = TOUR["active"]
    if tour_was_on:
        tour_stop("spin 180")
    with POLICY_LOCK:
        policy_was_on = POLICY["active"]
    if policy_was_on:
        policy_stop("spin 180")
    print("spin: 180° turn starting", flush=True)
    apply_command("right", 100)  # spin in place
    start = time.time()
    cancelled = False
    while time.time() - start < SPIN_180_TIME:
        with SPIN_LOCK:
            if SPIN["cancel"]:
                cancelled = True
                break
        # Keep the watchdog fed for the whole spin.
        with LOCK:
            STATE["last_cmd_time"] = time.time()
        time.sleep(0.05)
    if not cancelled:
        apply_command("stop", 100)
    # (The pose thread credits the turn from the motor throttles,
    # including a partial spin if this was cancelled.)
    with SPIN_LOCK:
        SPIN["active"] = False
    print(f"spin: 180° turn done{' (cancelled)' if cancelled else ''}",
          flush=True)
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/debug"):
            stats = debug_stats()
            if stats is None:
                self._reply("no frame yet — open the drive page first",
                            code=503)
                return
            self._reply(stats)
            return
        if self.path.startswith("/lidar"):
            now = time.time()
            with LIDAR_LOCK:
                payload = {
                    "online": LIDAR["online"] and
                              now - LIDAR["updated"] < 1.0,
                    "dist_m": LIDAR["dist_m"],
                    "strength": LIDAR["strength"],
                    "stop_m": get_mindist(),
                    "error": LIDAR["error"],
                    "heading": HEADING["deg"],
                    "trail": [[h, d, round(now - t, 1)]
                              for h, d, t in TRAIL
                              if now - t < TRAIL_MAX_AGE],
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/ld19"):
            now = time.time()
            with LD19_LOCK:
                online = LD19["online"] and now - LD19["updated"] < 1.0
                payload = {
                    "online": online,
                    "port": LD19["port"],
                    "hz": round(LD19["hz"], 1),
                    "points": LD19["points"] if online else 0,
                    "error": LD19["error"],
                    # index = whole degree (0 = sensor front, clockwise),
                    # value = latest distance in metres, or null.
                    "scan": [None if v is None else round(v, 2)
                             for v in LD19["bins"]] if online else [],
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/map.png"):
            png = map_render_png()
            if png is None:
                self.send_error(503, "map unavailable (cv2/numpy missing?)")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(png)
            return
        if self.path.startswith("/map.json"):
            stats = map_stats()
            if stats is None:
                self.send_error(503, "map unavailable (numpy missing?)")
                return
            body = json.dumps(stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/video_feed"):
            self.handle_video()
            return
        if self.path.startswith("/favicon.ico"):
            self.send_response(204)  # no favicon — keep the console clean
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(404)

    def handle_video(self):
        if not CAM.acquire():
            self.send_error(503, "camera could not start - is plant-capture "
                            "still holding it? (sudoers rule may be missing)")
            return
        try:
            jpg, seq = CAM.wait_jpg(-1, timeout=3.0)
            if jpg is None:
                self.send_error(503, "camera could not start")
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_seq = seq
            while True:
                jpg, seq = CAM.wait_jpg(last_seq, timeout=2.0)
                if jpg is None or seq == last_seq:
                    break  # camera stopped
                last_seq = seq
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: "
                                     b"image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            CAM.release()

    def _reply(self, msg, code=200):
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith("/map/clear"):
            map_clear()
            self._reply("map cleared — pose reset to start")
            return
        if self.path.startswith("/drive"):
            q = parse_qs(urlparse(self.path).query)
            try:
                left = float(q.get("left", ["0"])[0])
                right = float(q.get("right", ["0"])[0])
            except ValueError:
                left = right = 0.0
            if not MOTORS_OK:
                self._reply("motors unavailable — check HAT/I2C on the Pi")
            else:
                apply_drive(left, right)
                self._reply(f"L {int(left * 100)}% R {int(right * 100)}%")
            return
        if self.path.startswith("/auto/"):
            action = self.path.split("/")[2].split("?")[0]
            if action == "on":
                if not MOTORS_OK:
                    self._reply("motors unavailable — check HAT/I2C on the Pi")
                    return
                with POLICY_LOCK:
                    policy_was_on = POLICY["active"]
                if policy_was_on:
                    policy_stop("auto on")
                with TOUR_LOCK:
                    tour_was_on = TOUR["active"]
                if tour_was_on:
                    tour_stop("auto on")
                with AUTO_LOCK:
                    if AUTO["active"]:
                        AUTO["last_ping"] = time.time()
                        self._reply(AUTO["status"])
                        return
                    AUTO["active"] = True
                    AUTO["last_ping"] = time.time()
                    AUTO["status"] = "following…"
                if not CAM.acquire():
                    auto_stop("camera could not start")
                    self._reply("camera could not start", code=503)
                    return
                with AUTO_LOCK:
                    AUTO["cam"] = True
                self._reply("line follow on")
            elif action == "off":
                auto_stop("auto off")
                self._reply("auto off")
            elif action == "ping":
                with AUTO_LOCK:
                    if AUTO["active"]:
                        AUTO["last_ping"] = time.time()
                        self._reply(AUTO["status"])
                    else:
                        self._reply(AUTO["status"])
            else:
                self.send_error(404)
            return
        if self.path.startswith("/safety/mindist"):
            # ?m=<meters> — pick the minimum safety distance (the
            # both-LiDAR override threshold for tour and go-to-goal).
            q = parse_qs(urlparse(self.path).query)
            try:
                m = float(q.get("m", [""])[0])
            except ValueError:
                self._reply("bad distance", code=400)
                return
            self._reply(f"min safety distance {set_mindist(m):.2f} m")
            return
        if self.path.startswith("/policy/"):
            action = self.path.split("/")[2].split("?")[0]
            if action == "goal":
                # ?x=<m>&y=<m> — set (or retarget) the goal, from a
                # tap on the map. Works whether the policy is on or
                # off; when on, the rover retargets live.
                q = parse_qs(urlparse(self.path).query)
                try:
                    gx = float(q.get("x", [""])[0])
                    gy = float(q.get("y", [""])[0])
                except ValueError:
                    self._reply("bad goal", code=400)
                    return
                with POLICY_LOCK:
                    POLICY["goal_x"] = gx
                    POLICY["goal_y"] = gy
                    POLICY["goal_virtual"] = False
                self._reply(f"goal set ({gx:.2f}, {gy:.2f}) m")
            elif action == "on":
                if not MOTORS_OK:
                    self._reply("motors unavailable — check HAT/I2C on the Pi")
                    return
                # Turn the other modes off first — the tour in
                # particular clears its virtual goal, so the check
                # below only sees a goal the user actually tapped.
                with AUTO_LOCK:
                    auto_was_on = AUTO["active"]
                if auto_was_on:
                    auto_stop("policy on")
                with TOUR_LOCK:
                    tour_was_on = TOUR["active"]
                if tour_was_on:
                    tour_stop("policy on")
                with POLICY_LOCK:
                    has_goal = POLICY["goal_x"] is not None
                if not has_goal:
                    self._reply("tap the map to set a goal first", code=400)
                    return
                if not policy_load_model():
                    with POLICY_LOCK:
                        err = POLICY["model_error"]
                    self._reply(err, code=503)
                    return
                with POLICY_LOCK:
                    if POLICY["active"]:
                        POLICY["last_ping"] = time.time()
                        self._reply(POLICY["status"])
                        return
                    POLICY["active"] = True
                    POLICY["last_ping"] = time.time()
                    POLICY["status"] = "driving to goal…"
                self._reply("policy on — driving to goal")
            elif action == "off":
                policy_stop("policy off")
                self._reply("policy off")
            elif action == "ping":
                with POLICY_LOCK:
                    if POLICY["active"]:
                        POLICY["last_ping"] = time.time()
                    self._reply(POLICY["status"])
            else:
                self.send_error(404)
            return
        if self.path.startswith("/tour/"):
            action = self.path.split("/")[2].split("?")[0]
            if action == "on":
                if not MOTORS_OK:
                    self._reply("motors unavailable — check HAT/I2C on the Pi")
                    return
                if not policy_load_model():
                    with POLICY_LOCK:
                        err = POLICY["model_error"]
                    self._reply(err, code=503)
                    return
                with AUTO_LOCK:
                    auto_was_on = AUTO["active"]
                if auto_was_on:
                    auto_stop("tour on")
                # Pick the first virtual goal now; if the LD19 has
                # no answer the tour can't run.
                if not tour_pick_goal():
                    self._reply("LD19 offline — tour needs the LiDAR",
                                code=503)
                    return
                with TOUR_LOCK:
                    if TOUR["active"]:
                        TOUR["last_ping"] = time.time()
                        self._reply(TOUR["status"])
                        return
                    TOUR["active"] = True
                    TOUR["last_ping"] = time.time()
                    TOUR["repick_at"] = time.time() + TOUR_REPICK_S
                    TOUR["status"] = "touring…"
                with POLICY_LOCK:
                    POLICY["tour"] = True
                    POLICY["active"] = True
                    POLICY["last_ping"] = time.time()
                    POLICY["status"] = "touring…"
                self._reply("tour on — following open space")
            elif action == "off":
                tour_stop("tour off")
                self._reply("tour off")
            elif action == "ping":
                # Forward the heartbeat to the policy loop too — the
                # page pings /tour/ping while touring, not /policy/ping.
                # Return the policy's live status (what the rover is
                # doing) while touring, else the tour's own status.
                with TOUR_LOCK:
                    if TOUR["active"]:
                        TOUR["last_ping"] = time.time()
                    tour_active = TOUR["active"]
                    tour_status = TOUR["status"]
                with POLICY_LOCK:
                    if POLICY["active"]:
                        POLICY["last_ping"] = time.time()
                    pol_status = POLICY["status"]
                self._reply(pol_status if tour_active else tour_status)
            else:
                self.send_error(404)
            return
        if self.path.startswith("/stabilize/"):
            action = self.path.split("/")[2].split("?")[0]
            if action == "on":
                STABILIZE["on"] = True
                STABILIZER.reset()
                self._reply("stabilization on")
            elif action == "off":
                STABILIZE["on"] = False
                self._reply("stabilization off")
            else:
                self.send_error(404)
            return
        if self.path.startswith("/spin180"):
            if not MOTORS_OK:
                self._reply("motors unavailable — check HAT/I2C on the Pi")
                return
            with SPIN_LOCK:
                already = SPIN["active"]
            if already:
                self._reply("already spinning")
                return
            threading.Thread(target=spin_180, daemon=True).start()
            self._reply("spinning 180°…")
            return
        if not self.path.startswith("/cmd/"):
            self.send_error(404)
            return
        cmd = self.path.split("/")[2].split("?")[0]
        # Any manual button press cancels a 180° spin mid-turn.
        with SPIN_LOCK:
            SPIN["cancel"] = True
        # Any manual button press takes over from auto / tour /
        # policy instantly.
        with AUTO_LOCK:
            auto_was_on = AUTO["active"]
        if auto_was_on:
            auto_stop("manual override")
        with POLICY_LOCK:
            policy_was_on = POLICY["active"]
        if policy_was_on:
            policy_stop("manual override")
        with TOUR_LOCK:
            tour_was_on = TOUR["active"]
        if tour_was_on:
            tour_stop("manual override")
        # Manual driving invalidates the radar trail's heading
        # estimate — drop it so the picture restarts clean.
        with LIDAR_LOCK:
            TRAIL.clear()
        speed = STATE["speed"] * 100
        if "speed=" in self.path:
            try:
                speed = int(self.path.split("speed=")[1].split("&")[0])
            except ValueError:
                pass
        if not MOTORS_OK:
            msg = "motors unavailable — check HAT/I2C on the Pi"
        elif cmd in ("forward", "back"):
            # Manual safety: slow down as an obstacle approaches,
            # stop before hitting it. The factor is re-checked on
            # every repeat (the page re-sends a held button every
            # 500 ms) and enforced between repeats by the
            # manual_safety_loop thread.
            factor = manual_speed_factor(cmd)
            if factor <= 0:
                apply_command("stop", 100)
                msg = f"{cmd} blocked — obstacle too close"
            elif apply_command(cmd, int(int(speed) * factor)):
                msg = f"{cmd} @ {int(int(speed) * factor)}%"
                # Holding a button = repeated commands aren't sent, so refresh
                # the watchdog deadline while the command is active.
                with LOCK:
                    STATE["last_cmd_time"] = time.time()
            else:
                msg = f"unknown command: {cmd}"
        elif apply_command(cmd, int(speed)):
            msg = f"{cmd} @ {int(speed)}%"
            # Holding a button = repeated commands aren't sent, so refresh
            # the watchdog deadline while the command is active.
            with LOCK:
                STATE["last_cmd_time"] = time.time()
        else:
            msg = f"unknown command: {cmd}"
        self._reply(msg)


if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=lidar_loop, daemon=True).start()
    threading.Thread(target=ld19_loop, daemon=True).start()
    threading.Thread(target=pose_loop, daemon=True).start()
    threading.Thread(target=map_update_loop, daemon=True).start()
    threading.Thread(target=auto_loop, daemon=True).start()
    threading.Thread(target=policy_loop, daemon=True).start()
    threading.Thread(target=manual_safety_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", 8001), Handler)
    print("Rover drive page at http://localhost:8001  (Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_all()
        print("\nMotors stopped. Bye.")

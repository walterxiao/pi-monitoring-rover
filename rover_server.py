#!/usr/bin/env python3
"""Rover drive server — fullscreen camera, circular D-pad, line following,
obstacle avoid.

Run on the Pi:
    python3 rover_server.py
Then on your phone (same Wi-Fi):
    http://<your-pi's-address>:8001

The camera video fills the screen. A circular D-pad sits in the lower
left corner, overlaid on the video: ▲ forward, ▼ reverse, ◀ ▶ spin in
place — ◀ is left wheels backward + right wheels forward (turn left),
▶ is the reverse (turn right). That's how a 4WD car turns, there's
no steering wheel. Hold a button to drive, let go to stop. Drive
speed is fixed at full (100%).
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

OBSTACLE AVOID: tap "🚧 Obstacle Avoid" and the rover explores on its
own, keeping away from obstacles. The TF-Luna LiDAR measures the
distance straight ahead about 100 times a second: clear (over 0.5 m)
it keeps driving forward; blocked (under 0.5 m) it spins toward the
clearer side, using the camera's floor check to pick the side; too
close (under 0.25 m) it backs up for space instead of turning. If the
LiDAR is offline the loop falls back to the camera-only floor-color
logic, so the mode still works with the sensor unplugged. Escape
moves are committed, never flip-flopped: it keeps turning one way (up
to a full turn) instead of switching left/right every frame, and if
turning doesn't clear the path it backs up for a second to get space
and tries turning again. If it's blocked on all sides for 10 seconds
straight it stops and says "stuck". Same safety as line following:
any D-pad button takes over instantly, and it stops if the page
loses contact for 3 seconds. The drive page also shows a bird's-eye
radar view (top right): the rover in the middle, recent distance
readings plotted around it. The radar is rough — there is no gyro or
position tracking, so the picture is only good for a few seconds
while the rover turns or sits still.

SIGNS: draw a sign on paper (an arrow, a shape, a symbol — anything
distinctive) and tape it where the rover should act. Drive the rover
up to it, tap "🪧 Record sign", and give it a name plus the
instruction it means (turn left, turn right, go straight, back up,
stop). The rover saves the sign's visual fingerprint (ORB keypoints,
not the photo itself) plus how big it looked in the frame, in a
signs/ folder on the Pi. Then tap "🪧 Follow signs": the rover drives
forward on its own, and when it sees a sign at about the recorded
size it runs that sign's instruction. Size is the trigger distance —
a sign only fires when it looks about as big as when you recorded
it, so the rover acts at the right spot, not from across the room.
Same safety as the other modes: any D-pad button cancels, and it
stops if the page loses contact for 3 seconds.

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

STATE = {"last_cmd_time": 0.0, "moving": False, "speed": 1.0}
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
  #radar { top: 178px; right: 12px; width: 200px; height: 200px;
           border-radius: 12px; background: rgba(0,0,0,0.55);
           pointer-events: none; }
  #autotoggle { top: 52px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #autotoggle.on { background: rgba(56,161,105,0.9); }
  #avoidtoggle { top: 94px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #avoidtoggle.on { background: rgba(56,161,105,0.9); }
  #signtoggle { top: 136px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #signtoggle.on { background: rgba(56,161,105,0.9); }
  #signrecordbtn { top: 178px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #signlistbtn { top: 220px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #sign { top: 52px; right: 12px; font-size: 16px; font-weight: 600;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); }
  #signspanel { top: 262px; left: 16px; font-size: 16px;
           background: rgba(0,0,0,0.65); border-radius: 12px;
           padding: 10px 12px; display: none; max-width: 70vw; }
  #signspanel button { margin-left: 8px; font-size: 14px;
           border-radius: 10px; border: none;
           background: rgba(197,48,48,0.85); color: #fff; padding: 2px 8px; }
  .panel { top: 50%; left: 50%; transform: translate(-50%, -50%);
           width: 86vw; max-width: 420px; max-height: 72vh; overflow: auto;
           background: rgba(20,24,32,0.96); border-radius: 14px;
           padding: 14px 16px; display: none; font-size: 16px; }
  .panel h3 { margin: 0 0 10px 0; font-size: 18px; }
  .panel input, .panel select { font-size: 16px; padding: 8px; margin: 4px 0;
           border-radius: 8px; border: 1px solid #4a5568; width: 100%;
           background: #2d3748; color: #fff; box-sizing: border-box; }
  .panel button { font-size: 16px; padding: 8px 14px; margin: 4px 4px 4px 0;
           border-radius: 12px; border: none;
           background: rgba(43,108,176,0.9); color: #fff; }
  .panel button.danger { background: rgba(197,48,48,0.85); }
  .panel button.ghost { background: rgba(74,85,104,0.9); }
  .panel .row { display: flex; gap: 6px; align-items: center; }
  .panel .row select { flex: 1; }
  #signshot { width: 100%; border-radius: 10px; margin: 4px 0; }
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
  <canvas id="radar" class="overlay" width="200" height="200"></canvas>
  <button id="autotoggle" class="overlay">\U0001F916 Line Follow: Off</button>
  <button id="avoidtoggle" class="overlay">\U0001F6A7 Obstacle Avoid: Off</button>
  <button id="signtoggle" class="overlay">\U0001FAA7 Follow signs: Off</button>
  <button id="signrecordbtn" class="overlay">\U0001FAA7 Record sign</button>
  <button id="signlistbtn" class="overlay">\U0001FAA7 Signs</button>
  <div id="sign" class="overlay">\U0001FAA7 &mdash;</div>
  <div id="signspanel" class="overlay"></div>
  <div id="capturepanel" class="overlay panel">
    <h3>\U0001FAA7 Record this sign</h3>
    <img id="signshot" alt="Current view">
    <input id="signname" type="text" placeholder="Sign name" maxlength="30">
    <div>Instruction:</div>
    <select id="signinstruction">
      <option value="left">Turn left</option>
      <option value="right">Turn right</option>
      <option value="straight">Go straight</option>
      <option value="back">Back up</option>
      <option value="stop">Stop</option>
    </select>
    <div>
      <button id="signsave">Save</button>
      <button id="signretake" class="ghost">Retake</button>
      <button id="signclose" class="ghost">Close</button>
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
      if (avoidOn) setAvoid(false);
      setSignFollow(false);
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
  // Obstacle-avoid toggle. Same heartbeat pattern as line follow.
  const avoidtoggle = document.getElementById('avoidtoggle');
  let avoidOn = false;
  let avoidTimer = null;
  function setAvoid(on) {
    avoidOn = on;
    avoidtoggle.textContent = '\U0001F6A7 Obstacle Avoid: ' + (on ? 'On' : 'Off');
    avoidtoggle.classList.toggle('on', on);
    if (on) {
      if (autoOn) setAuto(false);
      setSignFollow(false);
      fetch('/avoid/on', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
      avoidTimer = setInterval(() => {
        fetch('/avoid/ping', {method: 'POST'})
          .then(r => r.text()).then(t => status.textContent = t)
          .catch(() => { setAvoid(false); status.textContent = 'no connection'; });
      }, 1000);
    } else {
      if (avoidTimer) { clearInterval(avoidTimer); avoidTimer = null; }
      fetch('/avoid/off', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
    }
  }
  avoidtoggle.addEventListener('click', () => setAvoid(!avoidOn));
  // D-pad buttons: hold to move, let go to stop. Pressing one while
  // auto is on switches back to manual first.
  document.querySelectorAll('button[data-cmd]').forEach(b => {
    const cmd = b.dataset.cmd;
    b.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (autoOn) setAuto(false);
      if (avoidOn) setAvoid(false);
      setSignFollow(false);
      start(cmd);
    });
    b.addEventListener('pointerup', halt);
    b.addEventListener('pointerleave', halt);
    b.addEventListener('pointercancel', halt);
  });
  // Center button: spin 180° in place. One tap, the rover turns
  // around and stops by itself. Any D-pad button cancels it mid-spin.
  document.getElementById('spinbtn').addEventListener('click', () => {
    if (autoOn) setAuto(false);
    if (avoidOn) setAvoid(false);
    setSignFollow(false);
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
  // Signs: record the sign in view under a name + instruction, and
  // show the sign the rover thinks it sees (checked every 2 s).
  const signEl = document.getElementById('sign');
  const signsPanel = document.getElementById('signspanel');
  function refreshSign() {
    fetch('/sign/current')
      .then(r => r.text())
      .then(t => signEl.textContent = t ? '\U0001FAA7 ' + t : '\U0001FAA7 —')
      .catch(() => {});
  }
  setInterval(refreshSign, 2000);
  refreshSign();
  // Capture panel: shows a snapshot of the current view so you can
  // see exactly what you are recording, then names the sign and
  // picks its instruction.
  const capturePanel = document.getElementById('capturepanel');
  const signshot = document.getElementById('signshot');
  const signname = document.getElementById('signname');
  const signinstruction = document.getElementById('signinstruction');
  function loadSnapshot() {
    signshot.src = '/snapshot?' + Date.now();
  }
  document.getElementById('signrecordbtn').addEventListener('click', () => {
    capturePanel.style.display = 'block';
    signname.value = '';
    loadSnapshot();
  });
  document.getElementById('signretake').addEventListener('click', loadSnapshot);
  document.getElementById('signclose').addEventListener('click', () => {
    capturePanel.style.display = 'none';
  });
  document.getElementById('signsave').addEventListener('click', () => {
    const name = signname.value.trim();
    if (!name) { status.textContent = 'type a name for this sign first'; return; }
    fetch('/sign/save?name=' + encodeURIComponent(name) +
          '&instruction=' + encodeURIComponent(signinstruction.value),
          {method: 'POST'})
      .then(r => r.text()).then(t => {
        status.textContent = t;
        capturePanel.style.display = 'none';
        refreshSign();
      })
      .catch(() => status.textContent = 'no connection');
  });
  document.getElementById('signlistbtn').addEventListener('click', () => {
    if (signsPanel.style.display === 'block') {
      signsPanel.style.display = 'none';
      return;
    }
    fetch('/signs')
      .then(r => r.text())
      .then(t => {
        const lines = t.split('\\n').filter(Boolean);
        signsPanel.innerHTML = lines.length
          ? lines.map(n => {
              const name = n.substring(0, n.lastIndexOf(' →'));
              return '<div>' + n +
                ' <button data-name="' + name + '">✕</button></div>';
            }).join('')
          : 'No signs saved yet.';
        signsPanel.style.display = 'block';
        signsPanel.querySelectorAll('button[data-name]').forEach(b => {
          b.addEventListener('click', () => {
            fetch('/sign/delete?name=' + encodeURIComponent(b.dataset.name),
                  {method: 'POST'})
              .then(r => r.text()).then(t => {
                status.textContent = t;
                signsPanel.style.display = 'none';
                refreshSign();
              });
          });
        });
      })
      .catch(() => status.textContent = 'no connection');
  });
  // Follow-signs toggle. Same heartbeat pattern as line follow and
  // obstacle avoid: the rover drives forward and runs a sign's
  // instruction when it sees one at about the recorded size.
  const signtoggle = document.getElementById('signtoggle');
  let signOn = false;
  let signTimer = null;
  function setSignFollow(on) {
    if (on === signOn) return;
    signOn = on;
    signtoggle.textContent = '\U0001FAA7 Follow signs: ' + (on ? 'On' : 'Off');
    signtoggle.classList.toggle('on', on);
    if (on) {
      if (autoOn) setAuto(false);
      if (avoidOn) setAvoid(false);
      fetch('/sign/on', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
      signTimer = setInterval(() => {
        fetch('/sign/ping', {method: 'POST'})
          .then(r => r.text()).then(t => status.textContent = t)
          .catch(() => { setSignFollow(false); status.textContent = 'no connection'; });
      }, 1000);
    } else {
      if (signTimer) { clearInterval(signTimer); signTimer = null; }
      fetch('/sign/off', {method: 'POST'})
        .then(r => r.text()).then(t => status.textContent = t);
    }
  }
  signtoggle.addEventListener('click', () => setSignFollow(!signOn));
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
  function drawRadar(data) {
    const W = radar.width, H = radar.height, cx = W / 2, cy = H / 2;
    const R = 92;            // pixels to the 3 m ring
    rctx.clearRect(0, 0, W, H);
    rctx.strokeStyle = 'rgba(255,255,255,0.25)';
    rctx.lineWidth = 1;
    [1, 2, 3].forEach(m => {
      rctx.beginPath();
      rctx.arc(cx, cy, R * m / RADAR_MAX_M, 0, 2 * Math.PI);
      rctx.stroke();
    });
    rctx.fillStyle = 'rgba(255,255,255,0.5)';
    rctx.font = '10px sans-serif';
    rctx.fillText('1m', cx + 3, cy - R / 3 + 3);
    rctx.fillText('2m', cx + 3, cy - 2 * R / 3 + 3);
    rctx.fillText('3m', cx + 3, cy - R + 10);
    if (!data || !data.online) {
      rctx.fillStyle = '#fc8181';
      rctx.font = '13px sans-serif';
      rctx.textAlign = 'center';
      rctx.fillText('LiDAR offline', cx, cy + 30);
      rctx.font = '11px sans-serif';
      rctx.fillText((data && data.error) || 'no data', cx, cy + 48);
      rctx.textAlign = 'left';
    } else {
      const heading = data.heading || 0;
      (data.trail || []).forEach(pt => {
        // Heading grows counterclockwise (left turns add), so the
        // screen x flips: a point to the rover's left plots left.
        const rel = (pt[0] - heading) * Math.PI / 180;
        const d = Math.min(pt[1], RADAR_MAX_M);
        const r = R * d / RADAR_MAX_M;
        const x = cx - r * Math.sin(rel);
        const y = cy - r * Math.cos(rel);
        const alpha = Math.max(0.15, 1 - pt[2] / 8);
        rctx.fillStyle = 'rgba(252,211,77,' + alpha.toFixed(2) + ')';
        rctx.beginPath();
        rctx.arc(x, y, 3, 0, 2 * Math.PI);
        rctx.fill();
      });
      if (data.dist_m !== null && data.dist_m !== undefined) {
        const d = Math.min(data.dist_m, RADAR_MAX_M);
        const r = R * d / RADAR_MAX_M;
        rctx.fillStyle = '#48bb78';
        rctx.beginPath();
        rctx.arc(cx, cy - r, 5, 0, 2 * Math.PI);
        rctx.fill();
        rctx.fillStyle = '#fff';
        rctx.font = '13px sans-serif';
        rctx.textAlign = 'center';
        rctx.fillText(data.dist_m.toFixed(2) + ' m', cx, cy + 30);
        rctx.textAlign = 'left';
      }
    }
    rctx.fillStyle = '#4299e1';
    rctx.beginPath();
    rctx.moveTo(cx, cy - 10);
    rctx.lineTo(cx - 7, cy + 8);
    rctx.lineTo(cx + 7, cy + 8);
    rctx.closePath();
    rctx.fill();
  }
  function refreshRadar() {
    if (!radarOn) return;
    fetch('/lidar')
      .then(r => r.json())
      .then(drawRadar)
      .catch(() => drawRadar(null));
  }
  setInterval(refreshRadar, 250);
  refreshRadar();
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
        elif cmd == "back":
            set_motors(s, s)
        elif cmd == "left":      # spin in place: left wheels back,
            set_motors(s, -s)    # right wheels forward = turn left
        elif cmd == "right":     # spin in place: left wheels forward,
            set_motors(-s, s)    # right wheels back = turn right
        elif cmd == "stop":
            stop_all()
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
    """Draw the green center target box (and the last detected line
    position as a red dot) onto a copy of the frame for the video stream.
    The box only shows while line following is on. Detection itself
    always uses the raw frame."""
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
    if AVOID["active"]:
        # Look-ahead zone used by obstacle avoid (yellow, full width,
        # bottom third of the frame).
        ax0, ax1 = 0, w
        ay0, ay1 = int(h * 0.67), h
        cv2.rectangle(disp, (ax0, ay0), (ax1, ay1), (0, 255, 255), 3)
    if LAST_DETECT["found"]:
        cx = int(w / 2 + LAST_DETECT["offset"] * (w / 2))
        cv2.circle(disp, (cx, (y0 + y1) // 2), 10, (0, 0, 255), -1)
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


def floor_obstacle(frame):
    """Check the path ahead for obstacles by floor-color segmentation.
    Samples the floor color from the bottom band (closest floor), then
    checks the look-ahead zone for pixels that don't match the floor.
    The threshold adapts to the floor's own variation, so patterned
    floors don't false-trigger as easily. Returns (blocked, frac,
    left_frac, right_frac): frac is the non-floor share of the whole
    zone; left/right say which side is clearer for turning."""
    import numpy as np
    h, w = frame.shape[:2]
    band = frame[int(h * 0.88):int(h * 0.98)].astype(np.int16)
    ref = np.median(band.reshape(-1, 3), axis=0)
    band_diff = np.abs(band - ref).sum(axis=2)
    thresh = max(90.0, float(np.percentile(band_diff, 95)) * 1.5)
    x0, x1 = 0, w  # detection zone spans the full frame width
    y0, y1 = int(h * 0.67), h  # bottom third of the frame only
    zone = frame[y0:y1, x0:x1].astype(np.int16)
    diff = np.abs(zone - ref).sum(axis=2)
    nonfloor = diff > thresh
    frac = float(nonfloor.mean())
    half = nonfloor.shape[1] // 2
    left_frac = float(nonfloor[:, :half].mean())
    right_frac = float(nonfloor[:, half:].mean())
    return frac > AVOID_BLOCK_FRAC, frac, left_frac, right_frac


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
        f"avoid status: {AVOID['status']}",
        f"lidar: online={LIDAR['online']} dist={LIDAR['dist_m']} "
        f"strength={LIDAR['strength']} frames={LIDAR['frames']} "
        f"bad={LIDAR['bad']} err={LIDAR['error'] or 'none'}",
        f"heading: {HEADING['deg']:.0f} deg trail: {len(TRAIL)} pts",
        f"signs status: {SIGNS_STATE['status']}",
        f"saved signs: {', '.join(f'{n} → {i}' for n, i in SIGNS.list_signs()) or 'none'}",
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
# Heading estimate for the radar trail: degrees, counterclockwise
# positive (a left turn adds). Dead reckoning from commanded turns
# only — there is no gyro or encoder, so it drifts. The trail fades
# in seconds, so a rough heading is good enough for the picture.
HEADING = {"deg": 0.0}
TRAIL = []  # (heading_deg, dist_m, timestamp), pruned to the last 8 s
TRAIL_MAX_AGE = 8.0
# Turn rate at 100% power, from the SPIN_180_TIME calibration
# (180° in SPIN_180_TIME seconds).
TURN_RATE_DEG_S = 180.0 / SPIN_180_TIME


def heading_add(degrees):
    """Nudge the heading estimate (dead reckoning, drifts)."""
    with LIDAR_LOCK:
        HEADING["deg"] = (HEADING["deg"] + degrees) % 360.0


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
                        with LIDAR_LOCK:
                            LIDAR["online"] = True
                            LIDAR["dist_m"] = dist_m
                            LIDAR["strength"] = strength
                            LIDAR["temp_c"] = temp_c
                            LIDAR["updated"] = now
                            LIDAR["frames"] += 1
                            if now - last_sample >= 0.2:
                                last_sample = now
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
    sensor is offline or the reading is stale (>1 s old)."""
    with LIDAR_LOCK:
        if not LIDAR["online"]:
            return None
        if time.time() - LIDAR["updated"] > 1.0:
            return None
        return LIDAR["dist_m"], LIDAR["strength"]


AVOID = {"active": False, "cam": False, "last_ping": 0.0,
         "status": "avoid off"}
AVOID_LOCK = threading.Lock()
# Obstacle-avoid settings. The rover drives continuously at full
# power (low power can't move the loaded rover) — no stopping to
# scan; each loop checks the latest frame and either drives forward
# or spins toward the clearer side. Escape moves are COMMITTED: once
# it starts turning one way it keeps turning that way (up to a full
# turn) instead of flip-flopping; if turning doesn't clear the path
# it backs up for a second to get space, then tries turning again.
AVOID_BLOCK_FRAC = 0.05   # non-floor share of the zone that = blocked (sensitive: anything contrasting in the bottom third triggers a turn or back-up)
LIDAR_STOP_M = 0.5     # LiDAR: something closer than this = blocked, turn
LIDAR_CLOSE_M = 0.25   # LiDAR: closer than this = too close to turn, back up
AVOID_HEAVY_FRAC = 0.50   # non-floor share of the zone that = "a lot of obstacles" — still this blocked after turning means back up instead
AVOID_TURN_MIN = 0.5      # seconds of turning before the heavy check applies (enough to see a new direction)
AVOID_TURN_COMMIT = 3.0   # seconds to keep turning one way (~full turn)
AVOID_BACKUP_TIME = 1.0   # seconds to reverse when turning isn't working
AVOID_STUCK_TIME = 10.0   # seconds blocked in a row = stuck


def avoid_set_status(text):
    with AVOID_LOCK:
        AVOID["status"] = text


def avoid_stop(reason):
    """Stop avoid mode (and the motors) and give the camera back."""
    with AVOID_LOCK:
        was_active = AVOID["active"]
        AVOID["active"] = False
        AVOID["status"] = reason
        cam = AVOID["cam"]
        AVOID["cam"] = False
    apply_drive(0, 0)
    if cam:
        CAM.release()
    print(f"avoid: {reason}", flush=True)
    return was_active


def avoid_loop():
    """Obstacle-avoid control loop, rebuilt on the TF-Luna LiDAR.
    Each cycle (~10 Hz) it reads the distance straight ahead: clear
    (>= LIDAR_STOP_M) -> keep driving forward; blocked -> spin in
    place toward the clearer side (the camera's floor check picks
    the side). Too close (< LIDAR_CLOSE_M) -> back up for space
    instead of turning. If the LiDAR is offline or stale, the loop
    falls back to the old camera-only floor-color logic, so the
    mode still works with the sensor unplugged. Escape moves are
    COMMITTED, never flip-flopped: once it starts turning one way
    it keeps turning that way (up to a full turn); if that doesn't
    clear the path it backs up for a second to get space, then
    tries turning again. It never drives straight back into the
    same obstacle — a back-up always runs to completion and is
    followed by a turn, not by driving forward. If the path stays
    blocked for AVOID_STUCK_TIME straight, it stops and reports
    stuck. Runs while AVOID['active'] and the page heartbeat is alive."""
    blocked_since = None
    escape_side = None
    escape_since = None
    backup_until = 0.0
    while True:
        with AVOID_LOCK:
            active = AVOID["active"]
            last_ping = AVOID["last_ping"]
        if not active:
            time.sleep(0.1)
            continue
        if time.time() - last_ping > 3.0:
            avoid_stop("page gone — avoid stopped")
            time.sleep(0.1)
            continue
        now = time.time()
        # A back-up in progress runs to completion — don't let a
        # clearing frame cut it short and drive back into the obstacle.
        if now < backup_until:
            apply_drive(1.0, 1.0)  # reverse
            avoid_set_status("backing up")
            time.sleep(0.1)
            continue
        # --- sense: LiDAR first, camera as fallback + turn direction
        reading = lidar_reading()
        dist_m = None
        use_lidar = False
        if reading is not None:
            d, strength = reading
            # Rated range 0.2-8 m; strength 0 or 65535 = bad reading.
            if 0.2 <= d <= 8.0 and 0 < strength < 65535:
                dist_m = d
                use_lidar = True
        frame = CAM.get_frame()
        cam_blocked, frac, left_frac, right_frac = False, 0.0, 0.0, 0.0
        if frame is not None:
            cam_blocked, frac, left_frac, right_frac = floor_obstacle(frame)
        if frame is None and not use_lidar:
            # No sensor at all yet — wait, don't drive blind.
            time.sleep(0.05)
            continue
        if use_lidar:
            blocked = dist_m < LIDAR_STOP_M
            very_close = dist_m < LIDAR_CLOSE_M
            sensor = f"lidar {dist_m:.2f}m"
        else:
            blocked = cam_blocked
            very_close = False
            sensor = "camera (lidar offline)"
        if not blocked:
            blocked_since = None
            escape_side = None
            escape_since = None
            avoid_set_status(f"{sensor} — driving")
            apply_drive(-1.0, -1.0)
        else:
            if blocked_since is None:
                blocked_since = now
            if now - blocked_since > AVOID_STUCK_TIME:
                avoid_stop("stuck — blocked on all sides")
                continue
            if very_close:
                # Too close to spin safely: back up for space.
                backup_until = now + AVOID_BACKUP_TIME
                escape_side = None
                escape_since = None
                apply_drive(1.0, 1.0)
                avoid_set_status(f"{sensor} — too close, backing up")
                time.sleep(0.1)
                continue
            if escape_side is None:
                # Commit to the clearer side; don't re-pick every frame.
                escape_side = "left" if left_frac <= right_frac else "right"
                escape_since = now
            elif use_lidar and dist_m < LIDAR_CLOSE_M \
                    and now - escape_since > AVOID_TURN_MIN:
                # Turned a bit and still very close: back up for
                # space instead of keep turning into it.
                backup_until = now + AVOID_BACKUP_TIME
                escape_side = None
                escape_since = None
                apply_drive(1.0, 1.0)
                avoid_set_status(f"{sensor} — still close, backing up")
                time.sleep(0.1)
                continue
            elif not use_lidar and frac > AVOID_HEAVY_FRAC \
                    and now - escape_since > AVOID_TURN_MIN:
                # Camera fallback: turned a bit and the bottom third
                # is still mostly blocked: back up for space.
                backup_until = now + AVOID_BACKUP_TIME
                escape_side = None
                escape_since = None
                apply_drive(1.0, 1.0)
                avoid_set_status("still blocked — backing up")
                time.sleep(0.1)
                continue
            elif now - escape_since > AVOID_TURN_COMMIT:
                # A full turn this way didn't clear the path: back up
                # for a second to get space, then try turning again.
                backup_until = now + AVOID_BACKUP_TIME
                escape_side = None
                escape_since = None
                apply_drive(1.0, 1.0)
                avoid_set_status("backing up")
                time.sleep(0.1)
                continue
            avoid_set_status(f"{sensor} — turning {escape_side}")
            apply_command(escape_side, 100)
            # Dead-reckoned heading for the radar trail: this cycle
            # spins ~0.1 s at 100% power.
            heading_add(TURN_RATE_DEG_S * 0.1 if escape_side == "left"
                        else -TURN_RATE_DEG_S * 0.1)
        time.sleep(0.1)



# --- Signs ---------------------------------------------------------------
# A "sign" is a hand-drawn mark on paper (an arrow, a shape, a symbol)
# that means an instruction. Recording saves the sign's ORB keypoint
# fingerprint plus how big it looked in the frame (the keypoint
# bounding box), in a signs/ folder on the Pi. While "Follow signs"
# runs, the rover drives forward and watches for recorded signs; when
# one appears at about the recorded size it runs that sign's
# instruction. Size is the trigger distance: a sign only fires when
# it looks about as big as when it was recorded, so the rover acts
# at the right spot instead of from across the room.
SIGNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "signs")
SIGN_MATCH_MIN = 15    # good ORB matches to count as a seen sign (a
                       # hand-drawn sign has fewer keypoints than a room)
SIGN_SIZE_MIN = 0.7    # sign must appear at least 70% of recorded size
SIGN_SIZE_MAX = 1.5    # ...and at most 150% (too big = too close / wrong scale)
SIGN_COOLDOWN = 5.0    # seconds before the same sign can fire again
SIGN_DRIVE_SPEED = 0.7   # follow-signs driving speed (70% — 50% won't
                         # move the loaded rover; full power shakes the
                         # video; turns stay at 100% because their 90°
                         # timing was calibrated there)
SIGN_TURN_TIME = SPIN_180_TIME / 2  # 90° turn, timed like the route turns
SIGN_STRAIGHT_TIME = 2.0  # seconds of driving for a "go straight" sign
SIGN_BACK_TIME = 2.0      # seconds of driving for a "back up" sign
SIGN_INSTRUCTIONS = ("left", "right", "straight", "back", "stop")
SIGN_INSTRUCTION_LABELS = {
    "left": "Turn left", "right": "Turn right",
    "straight": "Go straight", "back": "Back up", "stop": "Stop",
}
SIGN_RECOG = {"current": None, "score": 0, "size": 0.0, "checked": 0.0}


def _sign_name_ok(name):
    """Sanitize a sign name for use as a filename."""
    name = (name or "").strip().replace("/", "_").replace("\\", "_")
    return name[:30] if name else "sign"


class SignStore:
    """Saved signs: ORB descriptors + keypoint positions + the size
    the sign appeared at when recorded (keypoint bounding box) +
    the instruction it means. Stored as signs/<name>.npz plus a
    <name>.jpg thumbnail of the view. No photo leaves the Pi."""

    def __init__(self):
        self.dir = SIGNS_DIR
        os.makedirs(self.dir, exist_ok=True)
        self.orb = cv2.ORB_create(nfeatures=500) if CV2_OK else None
        self.lock = threading.Lock()

    def _load(self, name):
        import numpy as np
        path = os.path.join(self.dir, name + ".npz")
        if not os.path.exists(path):
            return None
        try:
            d = np.load(path, allow_pickle=True)
            return {
                "des": d["des"],
                "pts": d["pts"],
                "instruction": str(d["instruction"]),
                "bbox": d["bbox"],
            }
        except Exception:
            return None

    def save(self, name, frame, instruction):
        import numpy as np
        if not CV2_OK or self.orb is None:
            return False, "OpenCV unavailable — can't record signs"
        if instruction not in SIGN_INSTRUCTIONS:
            return False, "unknown instruction"
        name = _sign_name_ok(name)
        small = cv2.resize(frame, (640, 360))
        kp, des = self.orb.detectAndCompute(small, None)
        if des is None or len(kp) < 10:
            return False, "not enough detail in this view — move closer or draw bolder"
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        path = os.path.join(self.dir, name + ".npz")
        with self.lock:
            np.savez(path, des=des, pts=pts,
                     instruction=np.array(instruction),
                     bbox=np.array([w, h], dtype=np.float32))
            cv2.imwrite(os.path.join(self.dir, name + ".jpg"), frame)
        return True, f'saved sign "{name}" → {SIGN_INSTRUCTION_LABELS[instruction]}'

    def list_signs(self):
        out = []
        try:
            names = sorted(f[:-4] for f in os.listdir(self.dir)
                           if f.endswith(".npz"))
        except Exception:
            return out
        for n in names:
            s = self._load(n)
            if s is not None:
                out.append((n, s["instruction"]))
        return out

    def delete(self, name):
        name = _sign_name_ok(name)
        removed = False
        for suffix in (".npz", ".jpg"):
            p = os.path.join(self.dir, name + suffix)
            if os.path.exists(p):
                try:
                    os.remove(p)
                    removed = True
                except Exception:
                    pass
        return removed

    def recognize(self, frame):
        """Best matching sign in this frame, or (None, 0, 0.0, 0.0).
        Returns (name, good_matches, size_ratio, cx): size_ratio is
        how big the sign looks now vs when recorded (1.0 = same size);
        cx is the centroid's horizontal offset (-1..1). A sign only
        counts when it has enough good matches AND its size is within
        [SIGN_SIZE_MIN, SIGN_SIZE_MAX] of the recorded size."""
        import numpy as np
        if not CV2_OK or self.orb is None:
            return None, 0, 0.0, 0.0
        small = cv2.resize(frame, (640, 360))
        kp, des = self.orb.detectAndCompute(small, None)
        if des is None or len(kp) < 5:
            return None, 0, 0.0, 0.0
        pts = np.array([k.pt for k in kp], dtype=np.float32)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            names = sorted(f[:-4] for f in os.listdir(self.dir)
                           if f.endswith(".npz"))
        except Exception:
            return None, 0, 0.0, 0.0
        best = (None, 0, 0.0, 0.0)
        for name in names:
            s = self._load(name)
            if s is None or s["des"] is None or len(s["des"]) < 5:
                continue
            try:
                matches = bf.knnMatch(des, s["des"], k=2)
            except Exception:
                continue
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    good.append(m[0])
            if len(good) < SIGN_MATCH_MIN:
                continue
            # Size: compare the spread of the MATCHED keypoints now
            # vs the same keypoints when recorded (same subset both
            # sides, so the comparison is fair).
            cur = pts[[m.queryIdx for m in good]]
            rec = s["pts"][[m.trainIdx for m in good]]
            cw = float(cur[:, 0].max() - cur[:, 0].min())
            ch = float(cur[:, 1].max() - cur[:, 1].min())
            rw = float(rec[:, 0].max() - rec[:, 0].min())
            rh = float(rec[:, 1].max() - rec[:, 1].min())
            if rw < 5 or rh < 5:
                size_ratio = 1.0  # recorded points too clustered to measure
            else:
                size_ratio = ((cw / rw) + (ch / rh)) / 2.0
            if not (SIGN_SIZE_MIN <= size_ratio <= SIGN_SIZE_MAX):
                continue
            score = len(good)
            if score > best[1]:
                cx = float(cur[:, 0].mean() / 640.0 * 2 - 1)
                best = (name, score, size_ratio, cx)
        return best


SIGNS = SignStore()
SIGNS_STATE = {"active": False, "last_ping": 0.0,
               "status": "signs off", "last_sign": None, "last_time": 0.0}
SIGNS_LOCK = threading.Lock()


def sign_set_status(text):
    with SIGNS_LOCK:
        SIGNS_STATE["status"] = text


def sign_stop(reason):
    with SIGNS_LOCK:
        was_active = SIGNS_STATE["active"]
        SIGNS_STATE["active"] = False
        SIGNS_STATE["status"] = reason
    if was_active:
        stop_all()
    print(f"signs: stopped ({reason})", flush=True)


def sign_start():
    with SIGNS_LOCK:
        if SIGNS_STATE["active"]:
            return False, "already following signs"
        if not SIGNS.list_signs():
            return False, "no signs recorded yet — record one first"
        SIGNS_STATE["active"] = True
        SIGNS_STATE["last_ping"] = time.time()
        SIGNS_STATE["status"] = "following signs…"
        SIGNS_STATE["last_sign"] = None
        SIGNS_STATE["last_time"] = 0.0
    return True, "following signs…"


def _sign_alive():
    with SIGNS_LOCK:
        return (SIGNS_STATE["active"]
                and time.time() - SIGNS_STATE["last_ping"] <= 3.0)


def _sign_timed_drive(left, right, seconds):
    """Drive at the given throttles for a fixed time (for sign
    instructions), stopping early if the mode is cancelled."""
    end = time.time() + seconds
    while time.time() < end:
        if not _sign_alive():
            break
        apply_drive(left, right)
        time.sleep(0.05)
    stop_all()


def _sign_timed_spin(side, seconds):
    """Spin in place for a fixed time (for a turn sign)."""
    start = time.time()
    end = start + seconds
    while time.time() < end:
        if not _sign_alive():
            break
        apply_command(side, 100)
        time.sleep(0.05)
    stop_all()
    # Dead-reckoned heading for the radar trail.
    heading_add(TURN_RATE_DEG_S * (time.time() - start) if side == "left"
                else -TURN_RATE_DEG_S * (time.time() - start))


def sign_loop():
    """Follow-signs control loop: drives forward and watches for
    recorded signs. When a sign appears at about the recorded size,
    it runs that sign's instruction (turn left/right, go straight,
    back up, or stop). The same sign can't fire again for
    SIGN_COOLDOWN seconds. If the camera has no frame yet it waits
    instead of driving blind. Same safety as the other modes: a
    D-pad press cancels, and it stops if the page loses contact
    for 3 seconds (the page heartbeats /sign/ping while it runs)."""
    while True:
        with SIGNS_LOCK:
            active = SIGNS_STATE["active"]
            last_ping = SIGNS_STATE["last_ping"]
        if not active:
            time.sleep(0.1)
            continue
        if time.time() - last_ping > 3.0:
            sign_stop("page gone — signs stopped")
            time.sleep(0.1)
            continue
        frame = CAM.get_frame()
        if frame is None:
            stop_all()
            time.sleep(0.1)
            continue
        name, score, size_ratio, _cx = SIGNS.recognize(frame)
        now = time.time()
        with SIGNS_LOCK:
            SIGN_RECOG["current"] = name
            SIGN_RECOG["score"] = score
            SIGN_RECOG["size"] = size_ratio
            SIGN_RECOG["checked"] = now
            last_sign = SIGNS_STATE["last_sign"]
            last_time = SIGNS_STATE["last_time"]
        if name is not None and (name != last_sign
                                 or now - last_time > SIGN_COOLDOWN):
            with SIGNS_LOCK:
                SIGNS_STATE["last_sign"] = name
                SIGNS_STATE["last_time"] = now
            instr = dict(SIGNS.list_signs()).get(name)
            label = SIGN_INSTRUCTION_LABELS.get(instr, instr)
            sign_set_status(f"saw {name} ({score} matches, {size_ratio:.2f}×) — {label}")
            if instr == "left":
                _sign_timed_spin("left", SIGN_TURN_TIME)
            elif instr == "right":
                _sign_timed_spin("right", SIGN_TURN_TIME)
            elif instr == "straight":
                _sign_timed_drive(-SIGN_DRIVE_SPEED, -SIGN_DRIVE_SPEED,
                                  SIGN_STRAIGHT_TIME)
            elif instr == "back":
                _sign_timed_drive(SIGN_DRIVE_SPEED, SIGN_DRIVE_SPEED,
                                  SIGN_BACK_TIME)
            elif instr == "stop":
                sign_stop(f"stopped at {name}")
                continue
            continue
        sign_set_status("following signs…"
                        + (f" last: {last_sign}" if last_sign else ""))
        apply_drive(-SIGN_DRIVE_SPEED, -SIGN_DRIVE_SPEED)
        time.sleep(0.1)


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
    # A manual move: stop auto / avoid first, like a D-pad press does.
    with AUTO_LOCK:
        auto_was_on = AUTO["active"]
    if auto_was_on:
        auto_stop("spin 180")
    with AVOID_LOCK:
        avoid_was_on = AVOID["active"]
    if avoid_was_on:
        avoid_stop("spin 180")
    with SIGNS_LOCK:
        signs_was_on = SIGNS_STATE["active"]
    if signs_was_on:
        sign_stop("spin 180")
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
        heading_add(-180.0)  # spun right for a half-turn
    else:
        # Partial spin: credit the time actually spent turning.
        heading_add(-TURN_RATE_DEG_S * (time.time() - start))
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
        if self.path.startswith("/signs"):
            lines = [f"{n} → {i}" for n, i in SIGNS.list_signs()]
            self._reply("\n".join(lines))
            return
        if self.path.startswith("/snapshot"):
            frame = CAM.get_frame()
            if frame is None:
                self._reply("no frame yet — open the drive page first",
                            code=503)
                return
            jpg = encode_jpeg(frame)
            if jpg is None:
                self._reply("could not capture a frame", code=503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return
        if self.path.startswith("/sign/current"):
            self._reply(SIGN_RECOG["current"] or "")
            return
        if self.path.startswith("/sign/active"):
            with SIGNS_LOCK:
                self._reply("1" if SIGNS_STATE["active"] else "0")
            return
        if self.path.startswith("/lidar"):
            now = time.time()
            with LIDAR_LOCK:
                payload = {
                    "online": LIDAR["online"] and
                              now - LIDAR["updated"] < 1.0,
                    "dist_m": LIDAR["dist_m"],
                    "strength": LIDAR["strength"],
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
                with AVOID_LOCK:
                    avoid_was_on = AVOID["active"]
                if avoid_was_on:
                    avoid_stop("auto on")
                with SIGNS_LOCK:
                    signs_was_on = SIGNS_STATE["active"]
                if signs_was_on:
                    sign_stop("auto on")
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
        if self.path.startswith("/avoid/"):
            action = self.path.split("/")[2].split("?")[0]
            if action == "on":
                if not MOTORS_OK:
                    self._reply("motors unavailable — check HAT/I2C on the Pi")
                    return
                with AUTO_LOCK:
                    auto_was_on = AUTO["active"]
                if auto_was_on:
                    auto_stop("avoid on")
                with SIGNS_LOCK:
                    signs_was_on = SIGNS_STATE["active"]
                if signs_was_on:
                    sign_stop("avoid on")
                with AVOID_LOCK:
                    if AVOID["active"]:
                        AVOID["last_ping"] = time.time()
                        self._reply(AVOID["status"])
                        return
                    AVOID["active"] = True
                    AVOID["last_ping"] = time.time()
                    AVOID["status"] = "scanning…"
                if not CAM.acquire():
                    avoid_stop("camera could not start")
                    self._reply("camera could not start", code=503)
                    return
                with AVOID_LOCK:
                    AVOID["cam"] = True
                self._reply("obstacle avoid on")
            elif action == "off":
                avoid_stop("avoid off")
                self._reply("avoid off")
            elif action == "ping":
                with AVOID_LOCK:
                    if AVOID["active"]:
                        AVOID["last_ping"] = time.time()
                    self._reply(AVOID["status"])
            else:
                self.send_error(404)
            return
        if self.path.startswith("/sign/"):
            action = self.path.split("/")[2].split("?")[0]
            q = parse_qs(urlparse(self.path).query)
            if action == "save":
                name = _sign_name_ok(q.get("name", [""])[0])
                instruction = (q.get("instruction", [""])[0] or "").strip()
                frame = CAM.get_frame()
                if frame is None:
                    self._reply("no frame yet — open the drive page first",
                                code=503)
                    return
                ok, msg = SIGNS.save(name, frame, instruction)
                self._reply(msg, code=200 if ok else 400)
            elif action == "delete":
                name = _sign_name_ok(q.get("name", [""])[0])
                if SIGNS.delete(name):
                    self._reply(f'deleted sign "{name}"')
                else:
                    self._reply(f'no sign named "{name}"', code=404)
            elif action == "on":
                if not MOTORS_OK:
                    self._reply("motors unavailable — check HAT/I2C on the Pi")
                    return
                ok, msg = sign_start()
                self._reply(msg, code=200 if ok else 400)
            elif action == "off":
                sign_stop("stopped")
                self._reply("signs off")
            elif action == "ping":
                with SIGNS_LOCK:
                    if SIGNS_STATE["active"]:
                        SIGNS_STATE["last_ping"] = time.time()
                    self._reply(SIGNS_STATE["status"])
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
        # Any manual button press takes over from auto / avoid / signs
        # instantly.
        with AUTO_LOCK:
            auto_was_on = AUTO["active"]
        if auto_was_on:
            auto_stop("manual override")
        with AVOID_LOCK:
            avoid_was_on = AVOID["active"]
        if avoid_was_on:
            avoid_stop("manual override")
        with SIGNS_LOCK:
            signs_was_on = SIGNS_STATE["active"]
        if signs_was_on:
            sign_stop("manual override")
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
    threading.Thread(target=auto_loop, daemon=True).start()
    threading.Thread(target=avoid_loop, daemon=True).start()
    threading.Thread(target=sign_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", 8001), Handler)
    print("Rover drive page at http://localhost:8001  (Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_all()
        print("\nMotors stopped. Bye.")

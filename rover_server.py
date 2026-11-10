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
own, keeping away from obstacles. It drives continuously — no stopping
to scan — checking the latest camera frame about 10 times a second
and comparing the path ahead against the floor color. If the way is
clear it keeps driving forward; if something is in the way it spins
toward the clearer side until the way clears. Escape moves are
committed, never flip-flopped: it keeps turning one way (up to a
full turn) instead of switching left/right every frame, and if
turning doesn't clear the path it backs up for a second to get space
and tries turning again. If it's blocked on all sides for 10 seconds
straight it stops and says "stuck". Same safety as line following:
any D-pad button takes over instantly, and it stops if the page
loses contact for 3 seconds. Works best on plain, evenly lit floors
— a patterned floor, a rug, or a dark shiny object can confuse it,
and it can't judge exact distance (it turns when an obstacle fills
the near part of the view).

PLACE MEMORY: tap "📍 Save place" and give the current view a name
("Kitchen", "Charging spot") — the rover saves the visual fingerprint
of what it sees (ORB keypoints, not the photo itself) in a places/
folder on the Pi. About once a second it compares the live view
against every saved place and shows the best match at the top of the
page ("📍 Kitchen"). It recognizes a place from a similar angle and
lighting; a big lighting change or approaching from the opposite
side can miss. "Go to a place" (driving toward a remembered view)
is a later step — this version only remembers and recognizes.

ROUTES ("GO TO A PLACE"): places chain together into routes. A route
is a destination plus steps; each step is a place to find plus a
direction to take there (turn left, turn right, follow, go straight).
Example: Kitchen -> Piano Room via [find Sofa, turn left] [find
Restroom, turn right] [follow Hallway]. The rover runs each step by
spinning to scan for the step's place, driving toward it until the
view matches strongly, doing the direction, then moving to the next
step; finally it finds the destination and stops there. Every step
starts with a full scan, so a sloppy timed turn can't throw off the
rest of the route. If a step's place can't be found after a full
spin, it stops and says which step failed. Turns are timed (a 90°
turn is about 0.75 s at full power, from the 180° spin), not
measured — an IMU would make them exact. Same safety as line
following: any D-pad button cancels, and it stops if the page
loses contact for 3 seconds. Routes live in routes.json on the Pi.

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
  #autotoggle { top: 52px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #autotoggle.on { background: rgba(56,161,105,0.9); }
  #avoidtoggle { top: 94px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #avoidtoggle.on { background: rgba(56,161,105,0.9); }
  #placetoggle { top: 136px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #placelistbtn { top: 178px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #routetoggle { top: 220px; left: 16px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none;
           background: rgba(45,55,72,0.8); color: #fff; }
  #gotostop { top: 94px; right: 12px; font-size: 16px; padding: 8px 14px;
           border-radius: 20px; border: none; display: none;
           background: rgba(197,48,48,0.9); color: #fff; }
  #place { top: 52px; right: 12px; font-size: 16px; font-weight: 600;
           text-shadow: 0 1px 4px rgba(0,0,0,0.8); }
  #placespanel { top: 262px; left: 16px; font-size: 16px;
           background: rgba(0,0,0,0.65); border-radius: 12px;
           padding: 10px 12px; display: none; max-width: 70vw; }
  #placespanel button { margin-left: 8px; font-size: 14px;
           border-radius: 10px; border: none;
           background: rgba(197,48,48,0.85); color: #fff; padding: 2px 8px; }
  #placespanel button[data-go] { background: rgba(43,108,176,0.9); }
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
  .panel .routeitem { padding: 8px 0;
           border-bottom: 1px solid rgba(255,255,255,0.12); }
  #placeshot { width: 100%; border-radius: 10px; margin: 4px 0; }
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
  <button id="autotoggle" class="overlay">\U0001F916 Line Follow: Off</button>
  <button id="avoidtoggle" class="overlay">\U0001F6A7 Obstacle Avoid: Off</button>
  <button id="placetoggle" class="overlay">\U0001F4CD Save place</button>
  <button id="placelistbtn" class="overlay">\U0001F4CD Places</button>
  <button id="routetoggle" class="overlay">\U0001F5FA Routes</button>
  <button id="gotostop" class="overlay">&#9632; Stop</button>
  <div id="place" class="overlay">\U0001F4CD &mdash;</div>
  <div id="placespanel" class="overlay"></div>
  <div id="capturepanel" class="overlay panel">
    <h3>\U0001F4CD Save this place</h3>
    <img id="placeshot" alt="Current view">
    <input id="placename" type="text" placeholder="Place name" maxlength="30">
    <div>
      <button id="placesave">Save</button>
      <button id="placeretake" class="ghost">Retake</button>
      <button id="placeclose" class="ghost">Close</button>
    </div>
  </div>
  <div id="routepanel" class="overlay panel">
    <h3>\U0001F5FA Routes</h3>
    <div id="routelist"></div>
    <h3 style="margin-top:14px">New route</h3>
    <input id="routename" type="text" placeholder="Route name" maxlength="30">
    <div>Destination:</div>
    <select id="routedest"></select>
    <div>Steps (place + direction):</div>
    <div id="routesteps"></div>
    <div>
      <button id="routeaddstep" class="ghost">+ Add step</button>
      <button id="routesave">Save route</button>
      <button id="routeclose" class="ghost">Close</button>
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
      showGotoStop(false);
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
      showGotoStop(false);
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
      showGotoStop(false);
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
    showGotoStop(false);
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
  // Place memory: save the current view under a name, and show the
  // place the rover thinks it is in (checked every 2 s).
  const placeEl = document.getElementById('place');
  const placesPanel = document.getElementById('placespanel');
  function refreshPlace() {
    fetch('/place/current')
      .then(r => r.text())
      .then(t => placeEl.textContent = t ? '\U0001F4CD ' + t : '\U0001F4CD —')
      .catch(() => {});
  }
  setInterval(refreshPlace, 2000);
  refreshPlace();
  // Capture panel: shows a snapshot of the current view so you can
  // see exactly what you are saving, then names the place.
  const capturePanel = document.getElementById('capturepanel');
  const placeshot = document.getElementById('placeshot');
  const placename = document.getElementById('placename');
  function loadSnapshot() {
    placeshot.src = '/snapshot?' + Date.now();
  }
  document.getElementById('placetoggle').addEventListener('click', () => {
    capturePanel.style.display = 'block';
    placename.value = '';
    loadSnapshot();
  });
  document.getElementById('placeretake').addEventListener('click', loadSnapshot);
  document.getElementById('placeclose').addEventListener('click', () => {
    capturePanel.style.display = 'none';
  });
  document.getElementById('placesave').addEventListener('click', () => {
    const name = placename.value.trim();
    if (!name) { status.textContent = 'type a name for this place first'; return; }
    fetch('/place/save?name=' + encodeURIComponent(name), {method: 'POST'})
      .then(r => r.text()).then(t => {
        status.textContent = t;
        capturePanel.style.display = 'none';
        refreshPlace();
      })
      .catch(() => status.textContent = 'no connection');
  });
  document.getElementById('placelistbtn').addEventListener('click', () => {
    if (placesPanel.style.display === 'block') {
      placesPanel.style.display = 'none';
      return;
    }
    fetch('/places')
      .then(r => r.text())
      .then(t => {
        const names = t.split('\\n').filter(Boolean);
        placesPanel.innerHTML = names.length
          ? names.map(n => {
              const name = n.substring(0, n.lastIndexOf(' ('));
              return '<div>' + n +
                ' <button data-go="' + name + '">Go</button>' +
                ' <button data-name="' + name + '">✕</button></div>';
            }).join('')
          : 'No places saved yet.';
        placesPanel.style.display = 'block';
        placesPanel.querySelectorAll('button[data-go]').forEach(b => {
          b.addEventListener('click', () => {
            placesPanel.style.display = 'none';
            startGotoPlace(b.dataset.go);
          });
        });
        placesPanel.querySelectorAll('button[data-name]').forEach(b => {
          b.addEventListener('click', () => {
            fetch('/place/delete?name=' + encodeURIComponent(b.dataset.name),
                  {method: 'POST'})
              .then(r => r.text()).then(t => {
                status.textContent = t;
                placesPanel.style.display = 'none';
                refreshPlace();
              });
          });
        });
      })
      .catch(() => status.textContent = 'no connection');
  });
  // Routes panel: list saved routes with Go, and a builder that
  // chains places with a direction per step.
  const routePanel = document.getElementById('routepanel');
  const routeList = document.getElementById('routelist');
  const routeDest = document.getElementById('routedest');
  const routeSteps = document.getElementById('routesteps');
  function placeNames(listText) {
    return listText.split('\\n').filter(Boolean).map(function(n) {
      return n.substring(0, n.lastIndexOf(' ('));
    });
  }
  function loadPlaceNames(cb) {
    fetch('/places').then(function(r) { return r.text(); })
      .then(function(t) { cb(placeNames(t)); })
      .catch(function() { cb([]); });
  }
  function fillPlaceSelects(names) {
    const opts = names.map(function(n) {
      return '<option value="' + n + '">' + n + '</option>';
    }).join('');
    const destCur = routeDest.value;
    routeDest.innerHTML = opts;
    if (destCur) routeDest.value = destCur;
    routeSteps.querySelectorAll('.step-place').forEach(function(s) {
      const cur = s.value;
      s.innerHTML = opts;
      if (cur) s.value = cur;
    });
  }
  function renderRoutes() {
    fetch('/routes').then(function(r) { return r.json(); })
      .then(function(data) {
        const routes = data.routes || [];
        routeList.innerHTML = routes.length ? routes.map(function(rt) {
          return '<div class="routeitem">' + rt.name + ' &rarr; ' +
            rt.destination + ' (' + rt.steps.length + ' steps)' +
            ' <button data-goto="' + rt.name + '">Go</button>' +
            ' <button class="danger" data-delroute="' + rt.name + '">✕</button></div>';
        }).join('') : 'No routes yet. Build one below.';
        routeList.querySelectorAll('[data-goto]').forEach(function(b) {
          b.addEventListener('click', function() {
            startGoto('/goto/route?name=' + encodeURIComponent(b.dataset.goto));
          });
        });
        routeList.querySelectorAll('[data-delroute]').forEach(function(b) {
          b.addEventListener('click', function() {
            fetch('/route/delete?name=' + encodeURIComponent(b.dataset.delroute),
                  {method: 'POST'})
              .then(function(r) { return r.text(); })
              .then(function(t) { status.textContent = t; renderRoutes(); });
          });
        });
      })
      .catch(function() { status.textContent = 'no connection'; });
  }
  function stepRowHTML() {
    return '<div class="row step">' +
      '<select class="step-place"></select>' +
      '<select class="step-action">' +
        '<option value="left">Turn left</option>' +
        '<option value="right">Turn right</option>' +
        '<option value="follow">Follow</option>' +
        '<option value="straight">Go straight</option>' +
      '</select>' +
      '<button class="danger step-del">✕</button></div>';
  }
  document.getElementById('routetoggle').addEventListener('click', () => {
    if (routePanel.style.display === 'block') {
      routePanel.style.display = 'none';
      return;
    }
    routePanel.style.display = 'block';
    renderRoutes();
    loadPlaceNames(fillPlaceSelects);
  });
  document.getElementById('routeclose').addEventListener('click', () => {
    routePanel.style.display = 'none';
  });
  document.getElementById('routeaddstep').addEventListener('click', () => {
    routeSteps.insertAdjacentHTML('beforeend', stepRowHTML());
    loadPlaceNames(fillPlaceSelects);
  });
  routeSteps.addEventListener('click', (e) => {
    if (e.target.classList.contains('step-del')) {
      e.target.closest('.step').remove();
    }
  });
  document.getElementById('routesave').addEventListener('click', () => {
    const name = document.getElementById('routename').value.trim();
    const dest = routeDest.value;
    const steps = [];
    routeSteps.querySelectorAll('.step').forEach(function(row) {
      steps.push({
        place: row.querySelector('.step-place').value,
        action: row.querySelector('.step-action').value
      });
    });
    if (!name) { status.textContent = 'give the route a name first'; return; }
    if (!dest) { status.textContent = 'pick a destination first'; return; }
    fetch('/route/save', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name, destination: dest, steps: steps})})
      .then(function(r) { return r.text(); })
      .then(function(t) {
        status.textContent = t;
        document.getElementById('routename').value = '';
        routeSteps.innerHTML = '';
        renderRoutes();
      })
      .catch(function() { status.textContent = 'no connection'; });
  });
  // Go-to: start a route (or a single place), heartbeat while it
  // runs, Stop button to cancel. Any D-pad press cancels too.
  const gotostop = document.getElementById('gotostop');
  let gotoTimer = null;
  function showGotoStop(on) {
    gotostop.style.display = on ? 'block' : 'none';
    if (!on && gotoTimer) { clearInterval(gotoTimer); gotoTimer = null; }
  }
  function startGoto(url) {
    if (autoOn) setAuto(false);
    if (avoidOn) setAvoid(false);
    fetch(url, {method: 'POST'})
      .then(function(r) { return r.text(); })
      .then(function(t) {
        status.textContent = t;
        if (t.indexOf('going to') === 0) {
          routePanel.style.display = 'none';
          placesPanel.style.display = 'none';
          showGotoStop(true);
          gotoTimer = setInterval(function() {
            fetch('/goto/ping', {method: 'POST'})
              .then(function(r) { return r.text(); })
              .then(function(s) {
                status.textContent = s;
                return fetch('/goto/active');
              })
              .then(function(r) { return r.text(); })
              .then(function(a) { if (a !== '1') showGotoStop(false); })
              .catch(function() {
                showGotoStop(false);
                status.textContent = 'no connection';
              });
          }, 1000);
        }
      })
      .catch(function() { status.textContent = 'no connection'; });
  }
  function startGotoPlace(name) {
    startGoto('/goto/place?name=' + encodeURIComponent(name));
  }
  gotostop.addEventListener('click', () => {
    fetch('/goto/off', {method: 'POST'})
      .then(function(r) { return r.text(); })
      .then(function(t) { status.textContent = t; showGotoStop(false); });
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
        # Look-ahead zone used by obstacle avoid (yellow, full width).
        ax0, ax1 = 0, w
        ay0, ay1 = int(h * 0.35), int(h * 0.92)
        cv2.rectangle(disp, (ax0, ay0), (ax1, ay1), (0, 255, 255), 3)
    if LAST_DETECT["found"]:
        cx = int(w / 2 + LAST_DETECT["offset"] * (w / 2))
        cv2.circle(disp, (cx, (y0 + y1) // 2), 10, (0, 0, 255), -1)
    return disp


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
            jpg = encode_jpeg(annotate_frame(frame))
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


# --- Place memory -------------------------------------------------------
# A "place" is a saved view: ORB keypoint descriptors from one frame,
# stored as places/<name>.npz (plus a <name>.jpg thumbnail of the view).
# Recognition compares the live frame against every saved place about
# once a second and reports the best match. No photo leaves the Pi.
PLACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "places")
# A place only counts as recognized with at least this many good
# ORB matches (Lowe ratio test). Real matches score in the hundreds;
# an unrelated view scored 19 in testing, so 30 leaves margin both
# ways. Raise it if the rover claims the wrong place; lower it if it
# misses a place it should know.
PLACE_MATCH_MIN = 30
PLACE_STATE = {"current": None, "score": 0, "checked": 0.0}


def _place_name_ok(name):
    """Clean a place name for use as a filename; '' if unusable."""
    name = re.sub(r"[^A-Za-z0-9 _\-]", "_", (name or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" _-")[:30]
    return name


class PlaceMemory:
    def __init__(self):
        self.orb = None
        self.matcher = None
        self.cache = {}  # name -> (mtime, descriptors, n_keypoints)
        if CV2_OK:
            self.orb = cv2.ORB_create(nfeatures=800)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _features(self, frame):
        """ORB keypoints + descriptors for a frame, downscaled to 640
        wide so saving and recognizing always see the same size."""
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        return small, kp, des

    def _load_all(self):
        """(Re)load any place files that changed since last read."""
        import numpy as np
        if not os.path.isdir(PLACES_DIR):
            return
        for fn in os.listdir(PLACES_DIR):
            if not fn.endswith(".npz"):
                continue
            name = fn[:-4]
            path = os.path.join(PLACES_DIR, fn)
            mtime = os.path.getmtime(path)
            cached = self.cache.get(name)
            if cached and cached[0] == mtime:
                continue
            try:
                data = np.load(path)
                des = data["des"]
                self.cache[name] = (mtime, des, int(data["n"]))
            except Exception:
                pass
        for name in list(self.cache):
            if not os.path.exists(os.path.join(PLACES_DIR, name + ".npz")):
                del self.cache[name]

    def save(self, name, frame):
        import numpy as np
        if not CV2_OK:
            return False, "OpenCV not available on this Pi"
        name = _place_name_ok(name)
        if not name:
            return False, "give the place a name (letters and numbers)"
        small, _, des = self._features(frame)
        if des is None or len(des) < 10:
            return False, "not enough detail in this view to remember"
        os.makedirs(PLACES_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(PLACES_DIR, name + ".npz"),
                            des=des, n=len(des))
        ok, jpg = cv2.imencode(".jpg", small,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with open(os.path.join(PLACES_DIR, name + ".jpg"), "wb") as f:
                f.write(jpg.tobytes())
        self.cache.pop(name, None)  # force a reload on next recognize
        return True, f'saved "{name}" ({len(des)} keypoints)'

    def list_places(self):
        self._load_all()
        return sorted((name, info[2]) for name, info in self.cache.items())

    def delete(self, name):
        name = _place_name_ok(name)
        removed = False
        for ext in (".npz", ".jpg"):
            path = os.path.join(PLACES_DIR, name + ext)
            if os.path.exists(path):
                os.remove(path)
                removed = True
        self.cache.pop(name, None)
        if PLACE_STATE["current"] == name:
            PLACE_STATE["current"] = None
            PLACE_STATE["score"] = 0
        return removed

    def recognize(self, frame):
        """Best matching saved place for this frame, or (None, score)."""
        name, score, _ = self.recognize_detail(frame)
        return name, score

    def recognize_detail(self, frame):
        """Best match plus where it sits in the frame. Returns
        (name, score, cx): cx in [-1, 1] is the horizontal center of
        the good matches, + = right of center, None when nothing
        recognized. Go-to steering uses cx to stay aimed at a place."""
        if not CV2_OK:
            return None, 0, None
        _, kp, des = self._features(frame)
        if des is None or kp is None:
            return None, 0, None
        self._load_all()
        best_name, best_score, best_xs = None, 0, None
        for name, (_, saved_des, _) in self.cache.items():
            if saved_des is None or len(saved_des) == 0:
                continue
            try:
                pairs = self.matcher.knnMatch(des, saved_des, k=2)
            except Exception:
                continue
            xs = []
            for pair in pairs:
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                    xs.append(kp[pair[0].queryIdx].pt[0])
            if len(xs) > best_score:
                best_name, best_score = name, len(xs)
                best_xs = xs
        if best_score >= PLACE_MATCH_MIN:
            cx = (sum(best_xs) / len(best_xs) - 320.0) / 320.0
            return best_name, best_score, max(-1.0, min(1.0, cx))
        return None, best_score, None


PLACES = PlaceMemory()


def place_loop():
    """Background: about once a second, check whether the live view
    matches a saved place. Only runs while frames are flowing (i.e.
    while the drive page is open)."""
    while True:
        time.sleep(1.0)
        frame = CAM.get_frame()
        if frame is None:
            continue
        try:
            name, score = PLACES.recognize(frame)
        except Exception:
            continue
        PLACE_STATE["current"] = name
        PLACE_STATE["score"] = score
        PLACE_STATE["checked"] = time.time()


# --- Routes -------------------------------------------------------------
# A route chains places with directions: {"name": ..., "destination":
# <place>, "steps": [{"place": ..., "action": "left"|"right"|"follow"|
# "straight"}]}. Saved as JSON in routes.json next to this file.
ROUTES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "routes.json")
ROUTES_LOCK = threading.Lock()
ROUTE_ACTIONS = ("left", "right", "follow", "straight")


def load_routes():
    try:
        with open(ROUTES_FILE) as f:
            data = json.load(f)
        routes = data.get("routes", []) if isinstance(data, dict) else []
        return [r for r in routes if isinstance(r, dict) and r.get("name")]
    except Exception:
        return []


def save_routes_file(routes):
    tmp = ROUTES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"routes": routes}, f, indent=1)
    os.replace(tmp, ROUTES_FILE)


def get_route(name):
    for r in load_routes():
        if r["name"] == name:
            return r
    return None


def save_route(data):
    """Validate and store a route dict. Returns (ok, message)."""
    if not isinstance(data, dict):
        return False, "bad route data"
    name = _place_name_ok(data.get("name", ""))
    if not name:
        return False, "give the route a name (letters and numbers)"
    known = {n for n, _ in PLACES.list_places()}
    dest = (data.get("destination") or "").strip()
    if dest not in known:
        return False, f'destination "{dest}" is not a saved place'
    steps = data.get("steps") or []
    clean_steps = []
    for s in steps:
        if not isinstance(s, dict):
            return False, "bad step in route"
        place = (s.get("place") or "").strip()
        action = (s.get("action") or "").strip()
        if place not in known:
            return False, f'step place "{place}" is not a saved place'
        if action not in ROUTE_ACTIONS:
            return False, f'bad direction "{action}"'
        clean_steps.append({"place": place, "action": action})
    route = {"name": name, "destination": dest, "steps": clean_steps}
    with ROUTES_LOCK:
        routes = [r for r in load_routes() if r["name"] != name]
        routes.append(route)
        save_routes_file(routes)
    return True, f'saved route "{name}" ({len(clean_steps)} steps)'


def delete_route(name):
    with ROUTES_LOCK:
        routes = load_routes()
        kept = [r for r in routes if r["name"] != name]
        if len(kept) == len(routes):
            return False
        save_routes_file(kept)
        return True


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
    y0, y1 = int(h * 0.35), int(h * 0.92)
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
    places = PLACES.list_places()
    current = PLACE_STATE["current"]
    place_line = (f"place: {current} ({PLACE_STATE['score']} matches)"
                  if current else
                  f"place: unknown (best {PLACE_STATE['score']} matches)")
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
        f"saved places: {', '.join(f'{n} ({k})' for n, k in places) or 'none'}",
        place_line,
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
AVOID_BLOCK_FRAC = 0.15   # non-floor share of the zone that = blocked
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
    """Obstacle-avoid control loop: drives continuously, no stopping
    to scan. Each cycle (~10 Hz) it checks the latest camera frame:
    path clear -> keep driving forward; blocked -> spin in place
    toward the clearer side. The rover is always moving. Escape
    moves are COMMITTED, never flip-flopped: once it starts turning
    one way it keeps turning that way (up to a full turn); if that
    doesn't clear the path it backs up for a second to get space,
    then tries turning again. It never drives straight back into the
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
        frame = CAM.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        blocked, frac, left_frac, right_frac = floor_obstacle(frame)
        if not blocked:
            blocked_since = None
            escape_side = None
            escape_since = None
            avoid_set_status(f"clear ({frac:.0%} ahead) — driving")
            apply_drive(-1.0, -1.0)
        else:
            if blocked_since is None:
                blocked_since = now
            if now - blocked_since > AVOID_STUCK_TIME:
                avoid_stop("stuck — blocked on all sides")
                continue
            if escape_side is None:
                # Commit to the clearer side; don't re-pick every frame.
                escape_side = "left" if left_frac <= right_frac else "right"
                escape_since = now
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
            avoid_set_status(f"blocked ({frac:.0%}) — turning {escape_side}")
            apply_command(escape_side, 100)
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
    with GOTO_LOCK:
        goto_was_on = GOTO["active"]
    if goto_was_on:
        goto_stop("spin 180")
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
    with SPIN_LOCK:
        SPIN["active"] = False
    print(f"spin: 180° turn done{' (cancelled)' if cancelled else ''}",
          flush=True)
    return True


# --- Go to (routes) -----------------------------------------------------
# Runs one route in its own thread: for each step, scan for the step's
# place, drive toward it, do the step's direction, then move on; at
# the end find the destination and stop there. Same safety as line
# following: a D-pad press cancels, and it stops if the page loses
# contact for 3 seconds (the page heartbeats /goto/ping while a
# route runs). The watchdog also stops the motors if commands stop.
GOTO = {"active": False, "cam": False, "last_ping": 0.0,
        "status": "goto off", "route": None}
GOTO_LOCK = threading.Lock()
# A 90° turn is timed, not measured: half of the 180° spin time.
# Calibrate the same way as SPIN_180_TIME if turns over/undershoot.
TURN_90_TIME = SPIN_180_TIME / 2
GOTO_CENTER_TOL = 0.2    # |cx| <= this counts as aimed at the place
GOTO_ARRIVE_SCORE = 150  # good matches that = "here" (recognition is 30)
GOTO_APPROACH_TIMEOUT = 30.0  # seconds to reach a place before giving up
GOTO_FOLLOW_TIMEOUT = 20.0    # seconds of following before giving up
GOTO_STRAIGHT_TIME = 2.0      # seconds of driving for a "straight" step
GOTO_SEARCH_STEPS = 12        # 12 x ~30° spin bursts = a full turn
GOTO_SEARCH_SPIN = 0.25       # seconds of spin per scan step
GOTO_SEARCH_PAUSE = 0.35      # seconds to let a frame settle per step


def goto_set_status(text):
    with GOTO_LOCK:
        GOTO["status"] = text


def goto_stop(reason):
    """Stop a route run (and the motors) and give the camera back."""
    with GOTO_LOCK:
        was_active = GOTO["active"]
        GOTO["active"] = False
        GOTO["status"] = reason
        GOTO["route"] = None
        cam = GOTO["cam"]
        GOTO["cam"] = False
    apply_drive(0, 0)
    if cam:
        CAM.release()
    print(f"goto: {reason}", flush=True)
    return was_active


def goto_start(route):
    """Start running a route dict. Returns (ok, message)."""
    with AUTO_LOCK:
        auto_was_on = AUTO["active"]
    if auto_was_on:
        auto_stop("goto on")
    with AVOID_LOCK:
        avoid_was_on = AVOID["active"]
    if avoid_was_on:
        avoid_stop("goto on")
    with GOTO_LOCK:
        if GOTO["active"]:
            return False, "already going somewhere — stop it first"
        GOTO["active"] = True
        GOTO["last_ping"] = time.time()
        GOTO["route"] = route
        GOTO["status"] = f"going to {route['destination']}…"
    if not CAM.acquire():
        goto_stop("camera could not start")
        return False, "camera could not start"
    with GOTO_LOCK:
        GOTO["cam"] = True
    threading.Thread(target=_goto_run, args=(route,), daemon=True).start()
    return True, f"going to {route['destination']}"


def _goto_alive():
    """Still running and the page heartbeat is fresh."""
    with GOTO_LOCK:
        return GOTO["active"] and time.time() - GOTO["last_ping"] <= 3.0


def _goto_sleep(seconds):
    """Sleep, but wake early if the run was cancelled. False = stop."""
    end = time.time() + seconds
    while time.time() < end:
        if not _goto_alive():
            return False
        time.sleep(0.05)
    return True


def _goto_search(place):
    """Spin in place in ~30° steps until the place is in view.
    True = found (stop here, it is roughly ahead)."""
    for _ in range(GOTO_SEARCH_STEPS):
        if not _goto_alive():
            return False
        frame = CAM.get_frame()
        if frame is not None:
            name, score, _ = PLACES.recognize_detail(frame)
            if name == place and score >= PLACE_MATCH_MIN:
                return True
        apply_command("right", 100)
        if not _goto_sleep(GOTO_SEARCH_SPIN):
            return False
        apply_command("stop", 100)
        if not _goto_sleep(GOTO_SEARCH_PAUSE):
            return False
    return False


def _goto_approach(place):
    """Drive toward a place in view until the view matches strongly.
    Steers like line following: stop and spin toward the place while
    it is off-center, drive straight while it is centered. A blocked
    path spins toward the clearer side instead of driving. Returns
    True on arrival, False if the place is lost or time runs out."""
    arrive_hits = 0
    lost_since = None
    start = time.time()
    while time.time() - start < GOTO_APPROACH_TIMEOUT:
        if not _goto_alive():
            return False
        frame = CAM.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        blocked, frac, left_frac, right_frac = floor_obstacle(frame)
        if blocked:
            side = "left" if left_frac <= right_frac else "right"
            goto_set_status(f"blocked — turning {side}")
            apply_command(side, 100)
            if not _goto_sleep(0.3):
                return False
            continue
        name, score, cx = PLACES.recognize_detail(frame)
        if name != place:
            if lost_since is None:
                lost_since = time.time()
            if time.time() - lost_since > 3.0:
                return False
            apply_command("stop", 100)
            time.sleep(0.1)
            continue
        lost_since = None
        if score >= GOTO_ARRIVE_SCORE:
            arrive_hits += 1
            if arrive_hits >= 2:
                return True
        else:
            arrive_hits = 0
        if cx is None or abs(cx) <= GOTO_CENTER_TOL:
            goto_set_status(f"going to {place} • {score}")
            apply_drive(-1.0, -1.0)
        else:
            side = "left" if cx < 0 else "right"
            goto_set_status(f"centering on {place} • turning {side}")
            apply_command(side, 100)
        time.sleep(0.1)
    return False


def _goto_timed_spin(side, seconds):
    """Spin one way for a set time (a timed turn). False = cancelled."""
    apply_command(side, 100)
    ok = _goto_sleep(seconds)
    apply_command("stop", 100)
    return ok


def _goto_action(action, next_target):
    """Do a step's direction after arriving at its place. Returns
    True to move on, False if the run should stop."""
    if action == "left":
        return _goto_timed_spin("left", TURN_90_TIME)
    if action == "right":
        return _goto_timed_spin("right", TURN_90_TIME)
    if action == "straight":
        end = time.time() + GOTO_STRAIGHT_TIME
        while time.time() < end:
            if not _goto_alive():
                apply_command("stop", 100)
                return False
            apply_drive(-1.0, -1.0)
            time.sleep(0.1)
        apply_command("stop", 100)
        return True
    if action == "follow":
        # Drive forward until the next target comes into view (or the
        # path stays blocked too long). Obstacles detour like approach.
        start = time.time()
        while time.time() - start < GOTO_FOLLOW_TIMEOUT:
            if not _goto_alive():
                apply_command("stop", 100)
                return False
            frame = CAM.get_frame()
            if frame is not None:
                name, score, _ = PLACES.recognize_detail(frame)
                if name == next_target and score >= PLACE_MATCH_MIN:
                    apply_command("stop", 100)
                    return True
                blocked, frac, lf, rf = floor_obstacle(frame)
                if blocked:
                    side = "left" if lf <= rf else "right"
                    apply_command(side, 100)
                    if not _goto_sleep(0.3):
                        return False
                    continue
            apply_drive(-1.0, -1.0)
            time.sleep(0.1)
        apply_command("stop", 100)
        return False
    return False


def _goto_reach(place):
    """Find a place and drive to it, with one retry if it is lost on
    the way. True = arrived."""
    for _ in range(2):
        if not _goto_alive():
            return False
        if not _goto_search(place):
            return False
        if _goto_approach(place):
            return True
        goto_set_status(f"lost {place} — looking again…")
    return False


def _goto_run(route):
    """Run a whole route, step by step, then the destination."""
    steps = route.get("steps") or []
    dest = route["destination"]
    for i, step in enumerate(steps):
        goto_set_status(f"step {i + 1}/{len(steps)}: "
                        f"finding {step['place']}…")
        if not _goto_reach(step["place"]):
            if not _goto_alive():
                return  # cancelled — goto_stop already ran
            goto_stop(f"can't reach {step['place']} — stopped")
            return
        goto_set_status(f"at {step['place']} — {step['action']}")
        nxt = steps[i + 1]["place"] if i + 1 < len(steps) else dest
        if not _goto_action(step["action"], nxt):
            if not _goto_alive():
                return  # cancelled
            goto_stop(f"lost the way after {step['place']} — stopped")
            return
    goto_set_status(f"finding {dest}…")
    if not _goto_reach(dest):
        if not _goto_alive():
            return  # cancelled
        goto_stop(f"can't reach {dest} — stopped")
        return
    goto_stop(f"arrived at {dest}")


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
        if self.path.startswith("/places"):
            lines = [f"{n} ({k})" for n, k in PLACES.list_places()]
            self._reply("\n".join(lines))
            return
        if self.path.startswith("/routes"):
            self._reply(json.dumps({"routes": load_routes()}))
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
        if self.path.startswith("/place/current"):
            self._reply(PLACE_STATE["current"] or "")
            return
        if self.path.startswith("/goto/active"):
            with GOTO_LOCK:
                self._reply("1" if GOTO["active"] else "0")
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
                with GOTO_LOCK:
                    goto_was_on = GOTO["active"]
                if goto_was_on:
                    goto_stop("auto on")
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
                with GOTO_LOCK:
                    goto_was_on = GOTO["active"]
                if goto_was_on:
                    goto_stop("avoid on")
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
        if self.path.startswith("/route/save"):
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._reply("bad route data", code=400)
                return
            ok, msg = save_route(data)
            self._reply(msg, code=200 if ok else 400)
            return
        if self.path.startswith("/route/delete"):
            q = parse_qs(urlparse(self.path).query)
            name = _place_name_ok(q.get("name", [""])[0])
            if delete_route(name):
                self._reply(f'deleted route "{name}"')
            else:
                self._reply(f'no route named "{name}"', code=404)
            return
        if self.path.startswith("/goto/"):
            action = self.path.split("/")[2].split("?")[0]
            q = parse_qs(urlparse(self.path).query)
            if action == "route":
                route = get_route(_place_name_ok(q.get("name", [""])[0]))
                if route is None:
                    self._reply("no such route", code=404)
                    return
                ok, msg = goto_start(route)
                self._reply(msg, code=200 if ok else 400)
            elif action == "place":
                place = (q.get("name", [""])[0] or "").strip()
                known = {n for n, _ in PLACES.list_places()}
                if place not in known:
                    self._reply(f'"{place}" is not a saved place', code=404)
                    return
                route = {"name": f"Go to {place}", "destination": place,
                         "steps": []}
                ok, msg = goto_start(route)
                self._reply(msg, code=200 if ok else 400)
            elif action == "off":
                goto_stop("stopped")
                self._reply("goto off")
            elif action == "ping":
                with GOTO_LOCK:
                    if GOTO["active"]:
                        GOTO["last_ping"] = time.time()
                    self._reply(GOTO["status"])
            else:
                self.send_error(404)
            return
        if self.path.startswith("/place/"):
            action = self.path.split("/")[2].split("?")[0]
            q = parse_qs(urlparse(self.path).query)
            name = _place_name_ok(q.get("name", [""])[0])
            if action == "save":
                frame = CAM.get_frame()
                if frame is None:
                    self._reply("no frame yet — open the drive page first",
                                code=503)
                    return
                ok, msg = PLACES.save(name, frame)
                self._reply(msg, code=200 if ok else 400)
            elif action == "delete":
                if PLACES.delete(name):
                    self._reply(f'deleted "{name}"')
                else:
                    self._reply(f'no place named "{name}"', code=404)
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
        # Any manual button press takes over from auto / avoid / goto
        # instantly.
        with AUTO_LOCK:
            auto_was_on = AUTO["active"]
        if auto_was_on:
            auto_stop("manual override")
        with AVOID_LOCK:
            avoid_was_on = AVOID["active"]
        if avoid_was_on:
            avoid_stop("manual override")
        with GOTO_LOCK:
            goto_was_on = GOTO["active"]
        if goto_was_on:
            goto_stop("manual override")
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
    threading.Thread(target=auto_loop, daemon=True).start()
    threading.Thread(target=avoid_loop, daemon=True).start()
    threading.Thread(target=place_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", 8001), Handler)
    print("Rover drive page at http://localhost:8001  (Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_all()
        print("\nMotors stopped. Bye.")

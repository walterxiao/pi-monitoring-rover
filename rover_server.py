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
the floor just ahead and steers toward it, at full speed (100%). The video
shows a green center box: the rover drives straight while the line is
inside the box. If the line drifts out of the box, the rover stops and
spins in place toward the line until it's back inside, then drives on
(the red dot marks where the camera thinks the line is).
Safety: it stops the moment the line is lost (no creeping forward to
search), if you tap any
D-pad button (manual takes over instantly), or if the page closes or
loses contact for 3 seconds (the page heartbeats while auto is on).
A watchdog also stops the motors within ~1.5 seconds if commands stop
arriving, so the rover can never run away on its own.

OBSTACLE AVOID: tap "🚧 Obstacle Avoid" and the rover explores on its
own, keeping away from obstacles. It works in stop-scan-drive cycles:
it stops, waits for the camera to settle (a sharp frame, not a blurry
one), and checks the path ahead by comparing it against the floor
color. If the way is clear it drives a short burst forward; if
something is in the way it spins toward the clearer side and checks
again. If it's blocked on all sides after several turns it stops and
says "stuck". Same safety as line following: any D-pad button takes
over instantly, and it stops if the page loses contact for 3 seconds.
Works best on plain, evenly lit floors — a patterned floor, a rug, or
a dark shiny object can confuse it, and it can't judge exact distance
(it stops when an obstacle fills the near part of the view).

The camera is shared: one background thread captures frames, and both
the video stream and the line follower read the latest frame, so you
can watch it follow the line live.

The plant monitor normally holds the camera open, so opening the video
feed (or line following) pauses the plant capture service while the
camera is in use, and it resumes when the camera is released. (Needs a
sudoers rule so the server may stop/start plant-capture; see
BUILD_GUIDE.md.)

Motor map (matches BUILD_GUIDE.md):
    M1 = front-left   M2 = rear-left   M3 = front-right   M4 = rear-right

Same shape as plant-monitor's serve.py (stdlib http.server), on port
8001 so the plant dashboard (8000) keeps running alongside.
"""
import io
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
</script>
</body></html>
"""


def set_motors(left, right):
    """left/right: -1.0..1.0 applied to both motors on that side."""
    if kit is None:
        return
    kit.motor1.throttle = left    # front-left
    kit.motor2.throttle = left    # rear-left
    kit.motor3.throttle = right   # front-right
    kit.motor4.throttle = right   # rear-right


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
    Detection itself always uses the raw frame."""
    if not CV2_OK:
        return frame
    h, w = frame.shape[:2]
    disp = frame.copy()
    x0 = int(w * (0.5 - CENTER_BOX_HALF))
    x1 = int(w * (0.5 + CENTER_BOX_HALF))
    y0, y1 = int(h * 0.55), int(h * 0.95)
    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
    if AVOID["active"]:
        # Look-ahead zone used by obstacle avoid (yellow).
        ax0, ax1 = int(w * 0.30), int(w * 0.70)
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
                "AfMode": controls.AfModeEnum.Continuous,
                # Cap the frame time at 1/30s: indoors the camera would
                # otherwise use a slow shutter, and any movement smears.
                "FrameDurationLimits": (10000, 33333),
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
    x0, x1 = int(w * 0.30), int(w * 0.70)
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
SPIN_180_TIME = 2.0  # seconds at 100% power
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
# Obstacle-avoid settings. The rover moves in short bursts at full
# power (low power can't move the loaded rover), stopping to scan with
# a sharp frame between bursts.
AVOID_DRIVE_TIME = 0.5    # seconds of forward driving per scan cycle
AVOID_SETTLE_TIME = 0.3   # seconds stopped before a scan (blur settles)
AVOID_TURN_TIME = 0.5     # seconds of spin per avoidance turn
AVOID_BLOCK_FRAC = 0.15   # non-floor share of the zone that = blocked
AVOID_STUCK_TURNS = 6     # turns in a row without moving = stuck


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
    """Obstacle-avoid control loop: stop, let the camera settle, scan
    the path ahead, then drive a short burst or spin toward the clearer
    side. Runs while AVOID['active'] and the page heartbeat is alive."""
    turns = 0
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
        # Stop and let the camera settle so the scan uses a sharp frame.
        apply_drive(0, 0)
        time.sleep(AVOID_SETTLE_TIME)
        frame = CAM.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        blocked, frac, left_frac, right_frac = floor_obstacle(frame)
        if not blocked:
            turns = 0
            avoid_set_status(f"clear ({frac:.0%} ahead) — moving")
            apply_drive(-1.0, -1.0)
            time.sleep(AVOID_DRIVE_TIME)
            apply_drive(0, 0)
        else:
            turns += 1
            if turns >= AVOID_STUCK_TURNS:
                avoid_stop("stuck — blocked on all sides")
                continue
            side = "left" if left_frac <= right_frac else "right"
            avoid_set_status(f"blocked ({frac:.0%}) — turning {side}")
            apply_command(side, 100)
            time.sleep(AVOID_TURN_TIME)
            apply_drive(0, 0)


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
        if self.path.startswith("/video_feed"):
            self.handle_video()
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
        # Any manual button press takes over from auto / avoid instantly.
        with AUTO_LOCK:
            auto_was_on = AUTO["active"]
        if auto_was_on:
            auto_stop("manual override")
        with AVOID_LOCK:
            avoid_was_on = AVOID["active"]
        if avoid_was_on:
            avoid_stop("manual override")
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
    server = ThreadingHTTPServer(("0.0.0.0", 8001), Handler)
    print("Rover drive page at http://localhost:8001  (Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_all()
        print("\nMotors stopped. Bye.")

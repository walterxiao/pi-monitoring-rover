#!/usr/bin/env python3
"""Rover drive server — big hold-to-drive buttons on your phone.

Run on the Pi:
    python3 rover_server.py
Then on your phone (same Wi-Fi):
    http://<your-pi's-address>:8001

Hold a button to drive, let go to stop. A watchdog stops the motors
within ~1.5 seconds if commands stop arriving (Wi-Fi hiccup, phone
sleep), so the rover can never run away on its own.

Motor map (matches BUILD_GUIDE.md):
    M1 = front-left   M2 = rear-left   M3 = front-right   M4 = rear-right

Same shape as plant-monitor's serve.py (stdlib http.server), on port
8001 so the plant dashboard (8000) keeps running alongside.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from adafruit_motorkit import MotorKit
    kit = MotorKit()
    MOTORS_OK = True
except Exception as e:  # library missing or HAT not found — page still loads
    kit = None
    MOTORS_OK = False
    print(f"WARNING: motors unavailable ({e}). Page will load; buttons "
          "will report the problem. Check I2C is on and the HAT is seated.")

STATE = {"last_cmd_time": 0.0, "moving": False, "speed": 0.5}
LOCK = threading.Lock()

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1,
      maximum-scale=1, user-scalable=no">
<title>Pi Rover</title>
<style>
  body { font-family: -apple-system, sans-serif; text-align: center;
         background: #111; color: #eee; margin: 0; padding: 16px;
         -webkit-user-select: none; user-select: none;
         touch-action: none; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr;
          gap: 12px; max-width: 420px; margin: 20px auto; }
  button { font-size: 44px; padding: 26px 0; border-radius: 16px;
           border: none; background: #2b6cb0; color: #fff; }
  button:active { background: #4299e1; }
  .stop { grid-column: 1 / 4; background: #c53030; font-size: 28px;
          padding: 18px 0; }
  .empty { visibility: hidden; }
  input[type=range] { width: 80%; max-width: 380px; }
  #status { min-height: 1.4em; color: #a0aec0; }
</style></head>
<body>
  <h2>\U0001F916 Pi Rover</h2>
  <div class="grid">
    <div class="empty"></div>
    <button data-cmd="forward">&#9650;</button>
    <div class="empty"></div>
    <button data-cmd="left">&#9664;</button>
    <button data-cmd="stop" style="background:#4a5568">&#9632;</button>
    <button data-cmd="right">&#9654;</button>
    <div class="empty"></div>
    <button data-cmd="back">&#9660;</button>
    <div class="empty"></div>
    <button class="stop" data-cmd="stop">STOP</button>
  </div>
  <p>Speed <span id="spd">50</span>%</p>
  <input type="range" id="speed" min="20" max="100" value="50">
  <p id="status"></p>
<script>
  const status = document.getElementById('status');
  const speed = document.getElementById('speed');
  let timer = null;
  function send(cmd) {
    fetch('/cmd/' + cmd + '?speed=' + speed.value, {method: 'POST'})
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
  document.querySelectorAll('button').forEach(b => {
    const cmd = b.dataset.cmd;
    b.addEventListener('pointerdown', e => {
      e.preventDefault();
      if (cmd === 'stop') { halt(); } else { start(cmd); }
    });
    if (cmd !== 'stop') {
      b.addEventListener('pointerup', halt);
      b.addEventListener('pointerleave', halt);
      b.addEventListener('pointercancel', halt);
    }
  });
  speed.addEventListener('input', () =>
    document.getElementById('spd').textContent = speed.value);
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
        if cmd == "forward":
            set_motors(s, s)
        elif cmd == "back":
            set_motors(-s, -s)
        elif cmd == "left":      # spin in place
            set_motors(-s, s)
        elif cmd == "right":     # spin in place
            set_motors(s, -s)
        elif cmd == "stop":
            stop_all()
        else:
            return False
        STATE["moving"] = cmd != "stop"
    return True


def watchdog():
    # The phone page heartbeats every 500ms while a button is held.
    # No heartbeat for 1.5s = let go, phone asleep, or Wi-Fi lost -> stop.
    while True:
        time.sleep(0.25)
        with LOCK:
            if STATE["moving"] and time.time() - STATE["last_cmd_time"] > 1.5:
                stop_all()
                STATE["moving"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.startswith("/cmd/"):
            self.send_error(404)
            return
        cmd = self.path.split("/")[2].split("?")[0]
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
        body = msg.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", 8001), Handler)
    print("Rover drive page at http://localhost:8001  (Ctrl+C to stop)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_all()
        print("\nMotors stopped. Bye.")

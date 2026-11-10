# Pi Rover — Build Guide

A drive-around-the-house camera rover built from your plant-monitor Pi 5.
Phone-controlled, no soldering, no autonomy (v1). Plan on **60–90 minutes**,
working slowly. Everything here is plain screwdriver work.

Your Pi, camera, and plant dashboard stay untouched — the rover is an add-on.
The plant server keeps running on port 8000; the rover drive page runs on
port **8001**.

---

## On the table before you start

**You already own**

- Raspberry Pi 5 + ArduCam IMX708 camera (from the plant monitor)
- USB-C cable for the Pi

**From the parts list**

- YonPhsy 2-layer acrylic 4WD chassis kit ($20.99 on Amazon — battery
  box included; the XiaoR GEEK kit at $25.99 is the alternative, but it
  has no battery box)
- Adafruit DC & Stepper Motor HAT for Raspberry Pi — Mini Kit ($22.13
  on Amazon)
  (fully assembled, no soldering — verified on Adafruit's product page)
- Your Anker MagGo power bank (already owned — powers the Pi 5 over USB-C)

**One small buy**

- **4 AA batteries.** These go in the battery box that comes with the
  YonPhsy chassis and power the *motors only*. 4×AA = 6 volts, right in the
  HAT's 4.5–13.5 V motor range. (Only if you pick the XiaoR chassis
  instead: add a 4×AA battery holder with bare wire ends, ~$5–8, since
  that kit has no battery box.)

**Tools**

- The little screwdrivers in the chassis kit, plus one Phillips of your own
- A book or small box (to prop the rover up, wheels in the air, for testing)
- Painter's tape + a pen (for floor "stations" later)
- Velcro straps or double-sided tape (to hold the power bank down)

---

## The one big idea: two separate power systems

Keep these apart in your head and the whole build makes sense:

| Power source | Feeds | How |
|---|---|---|
| USB-C PD power bank | The Pi 5 (the computer) | USB-C cable into the Pi, same as usual |
| 4×AA battery box | The 4 motors | Two wires into the HAT's power terminals |

Motors are electrically noisy and hungry; computers hate that. Separate
batteries = no random Pi reboots when you hit the gas.

---

## Phase 1 — Build the rolling chassis (~25 min)

1. Lay out the kit: 2 chassis plates, 4 yellow gear motors, 4 wheels,
   screws/nuts, motor wires.
2. **Bottom plate first.** Screw one motor into each corner bracket.
   Motors lie flat, shafts pointing *outward*, one per wheel position.
   All 4 motor shafts should point away from the center of the car.
3. Push a wheel onto each motor shaft. It friction-fits — push straight,
   don't wiggle it on at an angle.
4. **Plug the motor wires in now, before the top plate goes on.** Each
   motor has 2 wires. Once the top plate is on, the motors are hard to
   reach. Coil the wire pairs loosely and poke them up through the plate
   holes. Label them with tape if you can:
   **FL** (front-left), **FR**, **RL** (rear-left), **RR**.
   "Front" = whichever end you decide the camera will face. Pick one now
   and stay consistent.
5. Fit the top plate on its standoffs. Don't fully tighten until Phase 3 —
   you may want to move the Pi mount once.

✅ Checkpoint: the car rolls freely in a straight line when you push it.

---

## Phase 2 — Wire the motors to the HAT (~10 min)

The HAT has 4 motor terminal blocks labeled **M1, M2, M3, M4**. Each takes
one motor's 2 wires (loosen the little screw, push the wire in, tighten).

| Terminal | Motor |
|---|---|
| M1 | Rear-right (RR) |
| M2 | Rear-left (RL) |
| M3 | Front-right (FR) |
| M4 | Front-left (FL) |

(This is the measured wiring on this build — found Nov 9 with
`test_motor_map.py`. The motors are wired rear pair / front pair, not
left pair / right pair, so don't "fix" the table back to the
left/right guess.)

**Don't stress about which of a motor's 2 wires goes in which hole.** If a
wheel spins the wrong way later, the fix is just swapping that motor's two
wires — the Phase 4 test finds this for you. No damage is possible here.

Then the battery box: its **red wire → the HAT's power terminal marked +**
(voltage in), **black wire → the terminal marked − (GND)**. The HAT has
polarity protection, but get it right anyway. Leave the batteries *out*
until Phase 4 — motors that jump while you're holding the car are no fun.

---

## Phase 3 — Pi, HAT, camera, power bank (~15 min)

1. Pi off, power bank unplugged. Push the included **lifter header** onto
   the Pi's 40 pins (the extra-tall black piece in the Adafruit box — it
   gives the HAT clearance over the Pi 5's ports), then push the **HAT
   onto the header**. It only fits one way; match the board edges to the Pi.
2. Bolt or tape the Pi to the **top plate, toward the front half** of the
   car. The camera ribbon cable is short, so the Pi needs to live near
   where the camera will sit.
3. **Camera**: mount it at the very front edge, lens facing forward and
   tilted slightly down — that's the plant-viewing angle. A small blob of
   reusable putty or a mini tripod clamp works for v1; get fancy later.
   Connect the ribbon cable to the Pi exactly as it is now in the plant
   setup (same slot, same orientation) and tape a small loop of the cable
   down so bumps don't tug it loose.
4. **Power bank**: Velcro-strap it to the top plate behind the Pi. Plug
   USB-C into the Pi. Weight at the back balances the camera at the front.
5. Final check before power: no loose screws rolling around, no wire
   touching a wheel, ribbon cable seated.

✅ Checkpoint: Pi boots off the power bank, camera still works
(run your usual plant dashboard check), and
`vcgencmd get_throttled` in a terminal prints `throttled=0x0`
(0x0 = power is clean; anything else, tell me the number).

---

## Phase 4 — Software: make one wheel turn (~15 min)

On the Pi, in a terminal:

```bash
# 1. Turn on I2C (the Pi talks to the HAT over these two pins)
sudo raspi-config nonint do_i2c 0
sudo reboot
```

After the reboot:

```bash
# 2. Install Adafruit's motor library (same style as your plant setup)
sudo pip3 install --break-system-packages adafruit-circuitpython-motorkit

# 3. Confirm the Pi can see the HAT — you want "60" in the grid
sudo apt install -y i2c-tools
i2cdetect -y 1
```

No `60` in the grid = the HAT isn't seated. Power off, push the HAT down
firmly, try again. (This is the #1 beginner snag and it's mechanical, not
you.)

Now put the AA batteries in the battery box, prop the rover on the book
(**all 4 wheels in the air**), and run `test_motors.py` (included with this
guide). It spins **one motor at a time**, slowly, and asks you after each:

- Which wheel moved? (should match the one it names)
- Did it spin "forward"? (top of the wheel moving toward the front of the car)

For any wheel that spins backwards, power off and **swap that motor's two
wires** in its terminal block. Re-run until all 4 pass.

✅ Checkpoint: all 4 wheels spin, right way around, wheels in the air.

---

## Phase 5 — Drive it from your phone (~10 min)

Run `rover_server.py` (included with this guide) on the Pi:

```bash
python3 rover_server.py
```

On your phone (same Wi-Fi), open:

```
http://<your-pi's-address>:8001
```

You'll get the ArduCam video fullscreen, with a circular D-pad at the
bottom left: ▲ forward, ▼ reverse, ◀ ▶ spin in place — ◀ is left
wheels backward + right wheels forward (turn left), ▶ is the reverse
(turn right). **Hold** a button to move, **let go to stop.** That's
how a 4WD car turns, there's no steering wheel. The button in the center of the D-pad (↻ 180°) spins
the rover a half-turn in place — tap it once and it turns around by
itself. Let go of the D-pad first, then tap it; any D-pad button
cancels the spin mid-turn. The spin is timed, not measured: if it
turns less than a half-turn, raise `SPIN_180_TIME` in
`rover_server.py` (seconds at full power, now 1.5); if it
overshoots, lower it. (It was 2.0 for a while, but that was set when
the motor map was wrong and the spin barely moved — with the map
fixed, 2.0 overshoots; 1.0 undershot.)

Under the hood it's the pattern you already know from `serve.py`
(Python's built-in web server), just on port 8001 so both can run at
once. Two built-in safety nets:

- Let go of a button → motors stop.
- If the page loses contact (Wi-Fi hiccup, phone sleeps), a watchdog
  stops the motors in about a second. The rover can never run away.

First drive: floor, big empty room. The page defaults to 100% speed now,
so pick an open room. If forward actually goes backward, *all four*
motors are flipped — easiest fix is swapping each pair, or tell me and
I'll flip it in code.

**Run it as a service (so it starts on boot, no terminal needed):**

```bash
sudo tee /etc/systemd/system/rover-server.service > /dev/null <<'EOF'
[Unit]
Description=Pi Rover drive server (phone control on port 8001)
After=network-online.target
Wants=network-online.target

[Service]
User=xiao
WorkingDirectory=/home/xiao/pi-monitoring-rover
ExecStart=/usr/bin/python3 /home/xiao/pi-monitoring-rover/rover_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable rover-server
sudo systemctl start rover-server
```

If the server is already running in a terminal, stop it first with
Ctrl+C, otherwise the port is taken. Check it with
`systemctl status rover-server`, watch the logs with
`journalctl -u rover-server -f`, stop it with
`sudo systemctl stop rover-server`.

**Live video:** the page shows the ArduCam video fullscreen, with the
D-pad overlaid on it. The plant monitor normally holds the camera open, so opening the video
pauses the plant capture service while you watch — it resumes when you
close the page or tap the 📷 Camera button to switch it off. For that
handover the server needs permission to stop/start plant-capture:

```bash
echo 'xiao ALL=(root) NOPASSWD: /usr/bin/systemctl stop plant-capture, /usr/bin/systemctl start plant-capture' | sudo tee /etc/sudoers.d/rover-camera
sudo chmod 440 /etc/sudoers.d/rover-camera
sudo visudo -cf /etc/sudoers.d/rover-camera
```

---

## Phase 6 — Stations: the part that makes it a plant tool

The growth timelapse compares photos taken from the *same* spot, so a
roving camera only works if it returns to the same spots:

1. Put a small painter's-tape **X** on the floor at each plant viewpoint.
2. Add a number on the tape: Station 1, 2, 3…
3. To take a "reading": drive to the X, line the rover's front wheels up
   on the tape, then take the photo (your existing capture flow still
   works — the camera doesn't care that its Pi moved).

Same spot + same angle = the timelapse math stays honest.

---

## Phase 7 — Line following: the rover drives itself

The rover can follow a blue tape line on the floor on its own — no remote
control needed while it's running.

**Set up the line:**
1. Lay blue painter's tape on the floor (the camera looks for blue —
   it won't work on a blue floor or rug). About 2–5 cm wide, gentle
   curves, no sharp corners.
2. **Tilt the camera down** so the floor about 30–50 cm ahead of the
   rover is in view. If the camera points at the horizon, it can't see
   the line — this is the step people miss.

**Run it:** on the drive page, tap **🤖 Line Follow**. The rover
drives forward at full speed (100%) and steers toward the line; you can
watch it live on the video. The video shows a **green center box** —
that's the target zone: the rover drives straight while the line is
inside the box. If the line drifts out of the box, the rover **stops
and spins in place** toward the line until it's back inside, then
drives on. The **red dot** marks where the camera thinks the line is,
so if the dot isn't on the tape, the camera is seeing something else. Tap the button again to stop, or tap any
D-pad button — manual control takes over instantly.

**Safety:** it stops by itself the moment the line disappears — it
does not creep forward to search — or if the page closes / loses
contact for 3 seconds. The watchdog
still applies, so the motors stop if anything goes wrong.

**If it won't follow:** check the camera tilt first, then the line —
too thin, too pale, or a blue floor/rug are the usual causes. Bright,
even room light helps; a line crossing a blue rug will confuse it.
If it keeps stopping with "line lost" even though the tape is right
there, open `http://<pi-address>:8001/debug`
on your phone (with the drive page open in another tab) — it shows how
many blue pixels the camera sees and the average colors, which tells
whether the tape is in view at all.

---

## Phase 8 — Obstacle avoid: the rover explores on its own

The rover can wander a room on its own and steer around obstacles —
no remote control needed while it's running. It has no depth sensor,
so it judges obstacles from the camera image alone.

**How it works:** it drives continuously — no stopping to scan — and
checks the latest camera frame about 10 times a second, comparing
the path ahead against the floor color — the floor just under the
camera is the reference. If the way ahead looks like floor, it keeps
driving forward; if a chunk of the path ahead doesn't look like
floor, it spins in place toward the clearer side until the way
clears. If it's blocked on all sides for 6 seconds straight it stops
and says "stuck".

**Run it:** on the drive page, tap **🚧 Obstacle Avoid**. The video
shows a **yellow box** — that's the look-ahead zone it checks. The
status line tells you what it's doing: `clear (3% ahead) — driving`,
`blocked (42%) — turning left`, or `stuck — blocked on all sides`.
Tap the button again to stop, or tap any D-pad button — manual control
takes over instantly. It won't run at the same time as Line Follow;
turning one on turns the other off.

**Set up:** works best on a plain, evenly lit hard floor with the
camera tilted down like Phase 7. Clear the room of small stuff first
— the camera can't see things right at the nose, and a low obstacle
can hide under the view. Keep the first run short and watch it.

**Limits, honestly:** it can't judge exact distance — it only knows
"the path ahead doesn't look like floor". A patterned floor, a rug,
a dark shiny object, or a shadow can confuse it (it may turn for no
reason, or miss a glass or low obstacle). If it turns too
often, the floor is probably too patterned for this method.

---

## Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| Nothing in `i2cdetect` | HAT not fully seated | Power off, press HAT down firmly, retry |
| One wheel backwards | That motor's wires swapped | Swap its 2 wires in the terminal block |
| Rover veers while "straight" | One motor slower / wire loose | Re-tighten terminals; code can trim speeds later |
| Pi reboots when motors start | Motors sharing Pi power | They must not — battery box feeds motors only |
| `throttled` isn't 0x0 | Power bank can't keep up | Try the bank's other USB-C port; shorter/thicker cable |
| Drive page loads but the D-pad does nothing | rover_server not running, or wrong port | Check the terminal window; it's port **8001** |
| Camera stopped working | Ribbon cable jogged loose | Reseat it, tape a strain-relief loop |

---

## What's deliberately NOT in v1

- No depth sensor — obstacle avoid judges from the camera image only
  (see Phase 8); it can't measure exact distance.
- No charging-while-driving — unplug, charge the bank, plug back in.
- The plant capture schedule still runs wherever the Pi happens to be
  parked. If a scheduled photo matters, park at a station before it fires.

Stuck anywhere? Tell me the phase number and what you see — a photo of the
wiring helps more than a description.

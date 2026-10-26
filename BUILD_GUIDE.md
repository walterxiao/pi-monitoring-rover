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
| M1 | Front-left (FL) |
| M2 | Rear-left (RL) |
| M3 | Front-right (FR) |
| M4 | Rear-right (RR) |

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

You'll get big hold-to-drive buttons: ▲ ▼ ◀ ▶ and STOP.
**Hold** a button to move, **let go to stop.** Left/right spin the rover
in place (left wheels backward, right wheels forward, or the reverse) —
that's how a 4WD car turns, there's no steering wheel.

Under the hood it's the pattern you already know from `serve.py`
(Python's built-in web server), just on port 8001 so both can run at
once. Two built-in safety nets:

- Let go of a button → motors stop.
- If the page loses contact (Wi-Fi hiccup, phone sleeps), a watchdog
  stops the motors in about a second. The rover can never run away.

First drive: floor, slowest speed (the page defaults to 50%), big empty
room. If forward actually goes backward, *all four* motors are flipped —
easiest fix is swapping each pair, or tell me and I'll flip it in code.

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

## Troubleshooting quick table

| Symptom | Likely cause | Fix |
|---|---|---|
| Nothing in `i2cdetect` | HAT not fully seated | Power off, press HAT down firmly, retry |
| One wheel backwards | That motor's wires swapped | Swap its 2 wires in the terminal block |
| Rover veers while "straight" | One motor slower / wire loose | Re-tighten terminals; code can trim speeds later |
| Pi reboots when motors start | Motors sharing Pi power | They must not — battery box feeds motors only |
| `throttled` isn't 0x0 | Power bank can't keep up | Try the bank's other USB-C port; shorter/thicker cable |
| Drive page loads but buttons do nothing | rover_server not running, or wrong port | Check the terminal window; it's port **8001** |
| Camera stopped working | Ribbon cable jogged loose | Reseat it, tape a strain-relief loop |

---

## What's deliberately NOT in v1

- No self-driving, no sensors, no obstacle avoidance — you are the driver.
- No charging-while-driving — unplug, charge the bank, plug back in.
- The plant capture schedule still runs wherever the Pi happens to be
  parked. If a scheduled photo matters, park at a station before it fires.

Stuck anywhere? Tell me the phase number and what you see — a photo of the
wiring helps more than a description.

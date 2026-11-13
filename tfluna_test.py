#!/usr/bin/env python3
"""TF-Luna distance test for the Pi rover — standalone reader, no rover code touched.

Wiring (youyeetoo kit, USB-TTL adapter included):
    TF-Luna TX  -> adapter RX
    TF-Luna RX  -> adapter TX
    TF-Luna GND -> adapter GND
    TF-Luna VCC -> adapter 5V        (sensor supply is 3.7-5.2 V)
    Adapter jumper set to 3.3V       (sensor logic is 3.3 V — do not use 5V logic)
Then plug the adapter into a Pi USB port; it shows up as /dev/ttyUSB0.

Protocol (Benewake manual; verified against Maker Portal + Aerospace Jam docs):
    115200 baud, 8 data bits, no parity, 1 stop bit.
    Sensor streams 9-byte frames at 100 Hz by default:
      [0]=0x59 [1]=0x59 [2]=Dist_L [3]=Dist_H (cm, little-endian)
      [4]=Strength_L [5]=Strength_H [6]=Temp_L [7]=Temp_H [8]=Checksum
      Temp (deg C) = (Temp_raw / 8.0) - 256.0
      Checksum = sum of bytes 0..7, low byte only.
    Range 0.2-8 m; readings under 0.2 m are unreliable.

Run on the Pi:
    pip3 install pyserial            (once; skip if already installed)
    ls /dev/ttyUSB*                  (confirm the adapter port)
    python3 tfluna_test.py           (prints ~2 readings/sec; Ctrl-C to stop)
    python3 tfluna_test.py --port /dev/ttyUSB1 --seconds 30

Sanity check: point the sensor at a flat wall ~1 m away and compare the
printed distance with a tape measure; then try ~2 m. Strength should be
in the hundreds/thousands on a normal wall; near 0 or 65535 means a bad
reading (too close, too far, or a dark/shiny surface).
"""
import argparse
import sys
import time


def _load_serial():
    try:
        import serial
        return serial
    except ImportError:
        print("pyserial is not installed. Run: pip3 install pyserial",
              file=sys.stderr)
        sys.exit(1)


HEADER = b"\x59\x59"


def read_frame(ser):
    """Sync to the 0x59 0x59 header, read one 9-byte frame, return
    (distance_m, strength, temp_c) or None if the checksum fails."""
    # Sync: read one byte at a time until we see 0x59 0x59.
    while True:
        b = ser.read(1)
        if not b:
            return None
        if b == b"\x59":
            b2 = ser.read(1)
            if b2 == b"\x59":
                break
    rest = ser.read(7)
    if len(rest) != 7:
        return None
    frame = HEADER + rest
    if (sum(frame[:8]) & 0xFF) != frame[8]:
        return None  # bad frame — drop it and resync
    dist_cm = frame[2] | (frame[3] << 8)
    strength = frame[4] | (frame[5] << 8)
    temp_raw = frame[6] | (frame[7] << 8)
    temp_c = (temp_raw / 8.0) - 256.0
    return dist_cm / 100.0, strength, temp_c


def main():
    ap = argparse.ArgumentParser(description="TF-Luna distance test")
    ap.add_argument("--port", default="/dev/ttyUSB0",
                    help="serial port (default /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=0,
                    help="run for N seconds (0 = until Ctrl-C)")
    args = ap.parse_args()
    serial = _load_serial()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Could not open {args.port}: {exc}", file=sys.stderr)
        print("Check the adapter is plugged in: ls /dev/ttyUSB*", file=sys.stderr)
        sys.exit(1)

    print(f"TF-Luna on {args.port} @ {args.baud} baud — Ctrl-C to stop")
    print("Distance (m) | Strength | Chip temp (C)")
    start = time.time()
    last_print = 0.0
    try:
        while True:
            if args.seconds and time.time() - start > args.seconds:
                break
            frame = read_frame(ser)
            if frame is None:
                continue
            now = time.time()
            if now - last_print >= 0.5:  # ~2 prints/sec; sensor runs at 100 Hz
                dist_m, strength, temp_c = frame
                flag = ""
                if dist_m < 0.2:
                    flag = "  <-- under 0.2 m, unreliable"
                elif dist_m > 8.0:
                    flag = "  <-- over 8 m, out of range"
                print(f"{dist_m:6.2f} m  | {strength:6d} | {temp_c:5.1f} C{flag}",
                      flush=True)
                last_print = now
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()

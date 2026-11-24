#!/usr/bin/env python3
"""LD19 spinning LiDAR reader test — standalone, does not touch rover_server.py.

Hardware: youyeetoo FHL-LD19 kit (LDROBOT LD19, 12 m) with its USB adapter.
The adapter is a USB-serial bridge, so on the Pi it shows up as a
/dev/ttyUSB* port. No driver install needed on Linux. The rover already has
a TF-Luna adapter on one ttyUSB port; this script auto-detects which port
is the LD19 by looking for valid LD19 packets, so no port juggling.
Use --port to override.

Protocol (LDROBOT LD19 datasheet v1.0 / DFRobot kit manual):
  UART 230400 8N1, one-way. Sensor streams 47-byte packets, no command needed.
  Packet: 0x54 | VerLen 0x2C (12 points) | Speed LE u16 (deg/s)
           | Start angle LE u16 (0.01 deg) | 12 x (distance LE u16 mm, intensity u8)
           | End angle LE u16 (0.01 deg) | Timestamp LE u16 (ms, wraps 30000)
           | CRC8 (poly 0x4D, init 0x00) over all preceding bytes.
  Angles interpolate linearly start -> end; add 360 if end < start (wrap).
  Coordinate: left-handed, sensor front = 0 deg, angle increases clockwise.
  PWM pin unconnected / Hi-Z = internal speed control, ~10 Hz default.
  Sensor: 38.59 x 38.59 x 33.50 mm, 47 g, 5 V (4.5-5.5), ~180 mA run / 300 mA start.
  Keep the optical window (the spinning head band) uncovered — nothing may
  sit in the scan plane or it blocks the beam.

Usage on the Pi:
  sudo apt install -y python3-serial   # once, only if pyserial is missing
                                        # (already installed if you ran the
                                        # Nov 14 rover build commands;
                                        # pip3 will NOT work on Pi OS Bookworm)
  python3 ld19_test.py --selftest  # no hardware: verifies parser + CRC
  python3 ld19_test.py             # 10 s capture, sector summary + wall check
                                   # auto-detects the LD19's USB port
  python3 ld19_test.py --dump  # also print every point of the first full revolution
  python3 ld19_test.py --selftest   # no hardware: verifies the parser + CRC
                                    # against the manufacturer's worked example

Sanity check: point the rover at a flat wall ~1 m away. The 0-deg sector
should read ~1000 mm. Compare with a tape measure, then try 2 m.
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    serial = None

HEADER = 0x54
VER_LEN = 0x2C          # 12 points per packet
POINTS_PER_PACK = 12
PACKET_LEN = 47         # 1+1+2+2+12*3+2+2+1
MAX_DISTANCE_MM = 12000


def _make_crc_table():
    """CRC-8, poly 0x4D, init 0x00, MSB-first (LDROBOT LD19 manual table)."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


CRC_TABLE = _make_crc_table()


def crc8(data):
    crc = 0
    for b in data:
        crc = CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


def parse_packet(buf):
    """Parse one 47-byte packet. Returns dict or None (with reason)."""
    if len(buf) != PACKET_LEN:
        return None, "length %d != %d" % (len(buf), PACKET_LEN)
    if buf[0] != HEADER:
        return None, "bad header 0x%02x" % buf[0]
    if buf[1] != VER_LEN:
        return None, "bad ver_len 0x%02x" % buf[1]
    if crc8(buf[:-1]) != buf[-1]:
        return None, "CRC mismatch (got 0x%02x, want 0x%02x)" % (
            buf[-1], crc8(buf[:-1]))
    speed = buf[2] | (buf[3] << 8)              # deg/s
    start = (buf[4] | (buf[5] << 8)) / 100.0   # deg
    end = (buf[42] | (buf[43] << 8)) / 100.0   # deg
    timestamp = buf[44] | (buf[45] << 8)        # ms
    if end < start:
        end += 360.0
    step = (end - start) / (POINTS_PER_PACK - 1)
    points = []
    for i in range(POINTS_PER_PACK):
        off = 6 + i * 3
        dist = buf[off] | (buf[off + 1] << 8)   # mm
        intensity = buf[off + 2]
        angle = (start + step * i) % 360.0
        points.append((angle, dist, intensity))
    return {
        "speed_dps": speed,
        "start_deg": start % 360.0,
        "end_deg": end % 360.0,
        "timestamp_ms": timestamp,
        "points": points,
    }, None


# Manufacturer's worked example (DFRobot LD19 manual, "Example" section).
# Hex string reconstructed from the manual's own field table; the OCR in the
# manual text garbles some nibbles (E5->ES, D5->DS, D0->DO, C5->CS, C0->CO,
# 1A->lA) — the values below are the corrected bytes. CRC byte is 0x50.
EXAMPLE_PACKET = bytes([
    0x54, 0x2C, 0x68, 0x08, 0xAB, 0x7E,
    0xE0, 0x00, 0xE4, 0xDC, 0x00, 0xE2, 0xD9, 0x00, 0xE5,
    0xD5, 0x00, 0xE3, 0xD3, 0x00, 0xE4, 0xD0, 0x00, 0xE9,
    0xCD, 0x00, 0xE4, 0xCA, 0x00, 0xE2, 0xC7, 0x00, 0xE9,
    0xC5, 0x00, 0xE5, 0xC2, 0x00, 0xE5, 0xC0, 0x00, 0xE5,
    0xBE, 0x82, 0x3A, 0x1A, 0x50,
])


def selftest():
    """Verify parser + CRC against the manufacturer's worked example."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (" — " + detail if detail else ""))
        ok = ok and cond

    check("CRC table head", CRC_TABLE[:4] == [0x00, 0x4D, 0x9A, 0xD7],
          "got %s" % [hex(x) for x in CRC_TABLE[:4]])
    check("example packet length", len(EXAMPLE_PACKET) == PACKET_LEN,
          "%d bytes" % len(EXAMPLE_PACKET))
    check("example CRC", crc8(EXAMPLE_PACKET[:-1]) == EXAMPLE_PACKET[-1],
          "computed 0x%02x, packet says 0x50" % crc8(EXAMPLE_PACKET[:-1]))
    pkt, err = parse_packet(EXAMPLE_PACKET)
    check("example parses", pkt is not None, err or "")
    if pkt:
        check("speed 2152 deg/s", pkt["speed_dps"] == 2152, str(pkt["speed_dps"]))
        check("start 324.27 deg", abs(pkt["start_deg"] - 324.27) < 0.01,
              str(pkt["start_deg"]))
        check("end 334.70 deg", abs(pkt["end_deg"] - 334.70) < 0.01,
              str(pkt["end_deg"]))
        check("point 1 = 224 mm / 228", pkt["points"][0][1:] == (224, 228),
              str(pkt["points"][0]))
        check("point 12 = 192 mm / 229", pkt["points"][11][1:] == (192, 229),
              str(pkt["points"][11]))
        # Note: the manual's own parsed table misprints point 12's intensity
        # as 234 (0xEA) and point 2's distance as 200 mm — the hex string and
        # the passing CRC (0x50) show 229 and 220 mm instead. Trust the CRC.
        bad = bytearray(EXAMPLE_PACKET)
        bad[10] ^= 0x01
        _, err2 = parse_packet(bytes(bad))
        check("corrupted byte fails CRC", err2 is not None and "CRC" in err2,
              err2 or "parsed clean — bad!")
    print("SELFTEST " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def find_ld19_port(candidates=None):
    """Find the LD19's serial port by its data signature.

    Opens each /dev/ttyUSB* port at 230400 baud and looks for valid LD19
    packets (0x54 header + CRC). The TF-Luna adapter streams 0x59 frames
    at 115200 baud, which never validate as LD19 packets, so the two
    sensors are told apart by what they send, not by port name.
    Returns the port path, or None if no LD19 stream is found.
    """
    import glob
    if serial is None:
        return None
    if candidates is None:
        candidates = sorted(glob.glob("/dev/ttyUSB*"))
    for port in candidates:
        try:
            ser = serial.Serial(port, 230400, timeout=0.2)
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
                while True:
                    i = buf.find(bytes([HEADER]))
                    if i < 0:
                        buf.clear()
                        break
                    if len(buf) - i < PACKET_LEN:
                        del buf[:i]
                        break
                    pkt, _err = parse_packet(bytes(buf[i:i + PACKET_LEN]))
                    del buf[:i + PACKET_LEN]
                    if pkt is not None:
                        good += 1
                        if good >= 3:
                            break
            if good >= 3:
                return port
        except Exception:
            pass
        finally:
            ser.close()
    return None


def run(port, seconds, dump):
    if serial is None:
        print("pyserial is not installed. Run: sudo apt install -y python3-serial")
        return 2
    if port is None:
        port = find_ld19_port()
        if port is None:
            print("No LD19 found on any /dev/ttyUSB* port.")
            print("Check the adapter is plugged in: ls /dev/ttyUSB*")
            return 2
        print("LD19 auto-detected on %s" % port)
    try:
        ser = serial.Serial(port, 230400, timeout=0.5)
    except Exception as e:
        print("Could not open %s: %s" % (port, e))
        print("Check the adapter is plugged in: ls /dev/ttyUSB*")
        return 2
    print("Reading %s at 230400 for %d s — spin the sensor up, it needs ~3 s." % (port, seconds))
    buf = bytearray()
    good = bad = 0
    speeds = []
    bins = {}          # whole degree -> closest distance seen (mm)
    dumped = False
    t_end = time.time() + seconds
    last_status = time.time()
    try:
        while time.time() < t_end:
            chunk = ser.read(4096)
            if chunk:
                buf.extend(chunk)
            while True:
                i = buf.find(bytes([HEADER]))
                if i < 0:
                    buf.clear()
                    break
                if len(buf) - i < PACKET_LEN:
                    del buf[:i]
                    break
                pkt, err = parse_packet(bytes(buf[i:i + PACKET_LEN]))
                del buf[:i + PACKET_LEN]
                if pkt is None:
                    bad += 1
                    if bad <= 3:
                        print("  bad packet: %s" % err)
                    continue
                good += 1
                speeds.append(pkt["speed_dps"])
                for angle, dist, _intensity in pkt["points"]:
                    if 0 < dist <= MAX_DISTANCE_MM:
                        d = int(angle) % 360
                        if d not in bins or dist < bins[d]:
                            bins[d] = dist
                if dump and not dumped and len(bins) >= 300:
                    dumped = True
                    print("\nFirst near-full revolution (angle: distance mm):")
                    for d in sorted(bins):
                        print("  %3d deg: %5d mm" % (d, bins[d]))
            if time.time() - last_status >= 2:
                last_status = time.time()
                hz = (sum(speeds) / len(speeds) / 360.0) if speeds else 0
                print("  %d good / %d bad packets, spin ~%.1f Hz, %d/360 deg covered"
                      % (good, bad, hz, len(bins)))
    finally:
        ser.close()

    print("\n— %d good packets, %d bad —" % (good, bad))
    if not speeds:
        print("No valid packets. Check wiring/power: the kit adapter needs 5 V, "
              "and the sensor spins up ~3 s after power-on.")
        return 1
    hz = sum(speeds) / len(speeds) / 360.0
    print("Spin rate ~%.1f Hz (expect ~10 Hz internal control)" % hz)
    print("Coverage: %d of 360 degrees saw a return" % len(bins))
    print("\nClosest return per 45-degree sector (0 deg = sensor front):")
    for s in range(0, 360, 45):
        vals = [bins[d] for d in range(s, s + 45) if d in bins]
        if vals:
            print("  %3d-%3d deg: min %5d mm, mean %5d mm (%d deg)"
                  % (s, s + 45, min(vals), sum(vals) // len(vals), len(vals)))
        else:
            print("  %3d-%3d deg: no return" % (s, s + 45))
    front = [bins[d] for d in list(range(355, 360)) + list(range(0, 6)) if d in bins]
    if front:
        print("\nStraight ahead (355-5 deg): closest %d mm — compare with a tape measure."
              % min(front))
    return 0


def main():
    ap = argparse.ArgumentParser(description="LD19 LiDAR reader test")
    ap.add_argument("--port", default=None,
                    help="serial port (default: auto-detect the LD19)")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--dump", action="store_true",
                    help="print every point of the first full revolution")
    ap.add_argument("--selftest", action="store_true",
                    help="verify parser/CRC offline, no hardware needed")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return run(args.port, args.seconds, args.dump)


if __name__ == "__main__":
    sys.exit(main())

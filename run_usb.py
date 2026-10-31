"""Start the USB-only tunnel and the viewer together; stop only our own children."""
import argparse
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def find_iproxy():
    found = shutil.which("iproxy")
    if found:
        return found
    for location in ("/opt/homebrew/bin/iproxy", "/usr/local/bin/iproxy"):
        if Path(location).is_file():
            return location
    return None


def tunnel_command(binary, port, udid=None):
    command = [binary, "-l", "-s", "127.0.0.1"]
    if udid:
        command.extend(["-u", udid])
    return command + [f"{port}:8765"]


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udid", help="Choose one phone when several are attached (Xcode Devices identifier)")
    parser.add_argument("--port", type=int, default=8765, help="Mac loopback tunnel port; phone remains 8765")
    parser.add_argument("--web-port", type=int, default=8080)
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.port, args.web_port)):
        parser.error("Ports must be between 1 and 65535")
    if args.port == args.web_port:
        parser.error("Tunnel and browser ports must differ")
    binary = find_iproxy()
    if not binary:
        parser.exit(1, "USB setup needs iproxy. In Terminal run: brew install libusbmuxd\nThen run Run_USB_Mac.command again. See START_HERE.md in the main folder.\n")
    # Never connect the viewer to an unrelated service already owning this port.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
    except OSError:
        parser.exit(1, f"Mac port {args.port} is occupied. Close the earlier USB viewer/tunnel, or use run_usb.py --port 8766.\n")
    tunnel = viewer = None
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    # Closing the Terminal window should also clean up our tunnel.
    signal.signal(signal.SIGTERM, interrupted)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupted)
    try:
        print(f"USB: Mac 127.0.0.1:{args.port} → iPhone loopback:8765", flush=True)
        print("Open Debris Capture on the unlocked iPhone, choose USB, and tap Start. Keep this window open.", flush=True)
        tunnel = subprocess.Popen(tunnel_command(binary, args.port, args.udid))
        # Check for immediate option/bind errors before starting the receiver.
        try:
            code = tunnel.wait(timeout=.4)
            raise RuntimeError(f"iproxy exited ({code}). Update libusbmuxd and check the data cable / trusted phone.")
        except subprocess.TimeoutExpired:
            pass
        viewer = subprocess.Popen([sys.executable, str(ROOT / "app.py"), "--transport", "usb", "--host", "127.0.0.1", "--port", str(args.port), "--web-port", str(args.web_port), "--open-browser"], cwd=ROOT)
        while viewer.poll() is None:
            if tunnel.poll() is not None:
                raise RuntimeError("USB tunnel stopped. Reconnect the phone, then launch Run_USB_Mac.command again.")
            time.sleep(.2)
        return viewer.returncode
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        stop_child(viewer)
        stop_child(tunnel)


if __name__ == "__main__":
    raise SystemExit(main())

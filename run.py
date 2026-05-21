"""VietGuard Dev Server — python run.py"""
import os, sys

# Fix Windows UTF-8
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

HOST = "127.0.0.1"
PORT = 5000

print("=" * 50)
print("VietGuard - Dev Server (waitress)")
print("=" * 50)

try:
    print("[INFO] Loading app & PhoBERT model...")
    import app as vg_app
    print("[OK]   Model loaded")
    print(f"[OK]   Routes: {len(list(vg_app.app.url_map.iter_rules()))}")
    print()
    print(f"  >> http://{HOST}:{PORT}")
    print(f"  >> CTRL+C to stop")
    print()

    from waitress import serve
    serve(vg_app.app, host=HOST, port=PORT, threads=4)

except KeyboardInterrupt:
    print("\n[Stopped]")
except Exception as e:
    import traceback
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

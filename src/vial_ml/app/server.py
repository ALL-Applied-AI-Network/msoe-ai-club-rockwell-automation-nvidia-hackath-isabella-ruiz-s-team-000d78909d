"""Vial inspection app: a local web server around the trained model.

    pip install flask opencv-python numpy            (plus torch torchvision pillow for the real model)

    python app/server.py --model runs/twin_v4/best.pt                         # upload photos in the browser
    python app/server.py --model best.pt --samples path/to/cap_seal/images    # step through a folder
    python app/server.py --model best.pt --watch path/to/incoming             # live: new photos as they land
    python app/server.py --model best.pt --camera 0                           # live: a camera on this computer
    python app/server.py --demo --samples path/to/cap_seal/images             # no model/torch: fake results

Then open http://localhost:8000 . Every inspection is saved in --out (default app_runs/):
images/<id>.jpg, results/<id>.json and history.csv.
"""
import argparse
import csv
import io
import json
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inspection import DemoPredictor, ModelPredictor, OrderOnlyPredictor, Pipeline  # noqa: E402
from order import OrderTracker  # noqa: E402
import findings  # noqa: E402
import report  # noqa: E402

PHOTO_EXT = (".png", ".jpg", ".jpeg", ".bmp")


class Store:
    """Inspections kept in memory (latest 500) and on disk."""

    def __init__(self, out_dir):
        self.out = os.path.abspath(out_dir)   # Flask resolves relative paths from app/, not the current folder
        for sub in ("images", "results"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        self.items, self.order = {}, []
        self.lock = threading.Lock()
        self.seq = 0
        self.history_path = os.path.join(out_dir, "history.csv")
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w", newline="") as f:
                csv.writer(f).writerow(["id", "time", "source", "name", "verdict", "defects", "missed",
                                        "fill_count", "threshold", "model", "seconds"])

    def add(self, result, bgr, source, name):
        with self.lock:
            self.seq += 1
            rid = f"{datetime.now():%Y%m%d_%H%M%S}_{self.seq:04d}"
        result.update(id=rid, source=source, name=name, time=datetime.now().isoformat(timespec="seconds"))
        cv2.imwrite(os.path.join(self.out, "images", rid + ".jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(self.out, "results", rid + ".json"), "w") as f:
            json.dump(result, f, indent=1)
        bad = [r["position"] for r in result["positions"] if r["status"] == "defect"]
        missed = [r["position"] for r in result["positions"] if r["status"] == "missed"]
        with self.lock:
            self.items[rid] = result
            self.order.append(rid)
            if len(self.order) > 500:
                self.items.pop(self.order.pop(0), None)
            with open(self.history_path, "a", newline="") as f:
                csv.writer(f).writerow([rid, result["time"], source, name, result["verdict"], " ".join(bad),
                                        " ".join(missed), (result.get("fill") or {}).get("count", ""),
                                        result["threshold"], result["model"], result["seconds"]])
        return result

    def get(self, rid):
        with self.lock:
            return self.items.get(rid)

    def summaries(self):
        with self.lock:
            return [summary(self.items[r]) for r in reversed(self.order)]


def summary(r):
    c = r.get("counts") or {}
    return {"id": r["id"], "time": r["time"], "name": r["name"], "source": r["source"], "verdict": r["verdict"],
            "defects": c.get("defect", 0), "missed": c.get("missed", 0), "seconds": r["seconds"]}


def build_app(args):
    if args.demo:
        predictor = DemoPredictor()
    elif args.order_only:
        predictor = OrderOnlyPredictor()
    else:
        predictor = ModelPredictor(args.model, threshold=args.threshold)
    pipe = Pipeline(predictor, align=not args.no_align)
    store = Store(args.out)
    run_lock = threading.Lock()
    settings = {"threshold": predictor.threshold, "fill_count": None}
    live = {"source": None, "latest": None, "error": None, "frames": 0}
    tracker = OrderTracker(pipe.names, pipe.pts)
    samples = sorted(f for f in os.listdir(args.samples) if f.lower().endswith(PHOTO_EXT)) if args.samples else []

    def inspect(bgr, source, name, ts=None):
        t0 = time.time()
        with run_lock:
            result = pipe.run(bgr, threshold=settings["threshold"], fill_count=settings["fill_count"])
            # Order tracking: compare with the empty-tray reference (only photos with the arm clear)
            if tracker.active and result["verdict"] != "SKIPPED":
                probs = None if getattr(predictor, "order_only", False) else result.get("probs")
                tracker.update(pipe.normalise(bgr), name, probs, ts=ts)
            result["order"] = tracker.summary() if tracker.active else None
            result["findings"] = (findings.from_order(result["order"]) if result["order"]
                                  else findings.from_photo(result))
        result["seconds"] = round(time.time() - t0, 3)
        return store.add(result, pipe.normalise(bgr), source, name)

    def decode(data):
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            abort(400, "could not read that file as an image")
        return bgr

    def view(result, threshold=None, fill=None):
        """A stored result, optionally re-scored with another threshold / fill count, ready for the UI."""
        if threshold is not None or fill is not None:
            thr = settings["threshold"] if threshold is None else threshold
            result = pipe.rescore(result, thr, fill)
        return {**result, "image_url": f"/api/image/{result['id']}"}

    # ---------- live sources ----------
    def watch_loop(folder):
        seen = set(f for f in os.listdir(folder)) if not args.include_existing else set()
        live["source"] = f"folder: {os.path.abspath(folder)}"
        while True:
            try:
                for f in sorted(os.listdir(folder)):
                    if f in seen or not f.lower().endswith(PHOTO_EXT):
                        continue
                    seen.add(f)
                    time.sleep(0.3)   # let the camera software finish writing
                    path = os.path.join(folder, f)
                    bgr = cv2.imread(path)
                    if bgr is not None:
                        # capture time: from the file name if it has one, else when the file was written
                        live["latest"] = inspect(bgr, "live", f, ts=os.path.getmtime(path))["id"]
                        live["frames"] += 1
                live["error"] = None
            except Exception as e:  # keep watching; show the error in the UI
                live["error"] = str(e)
            time.sleep(args.every)

    def camera_loop(index):
        live["source"] = f"camera {index}"
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ok, frame = cap.read()
            if not ok:
                live["error"] = f"camera {index} is not returning frames"
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(index)
                continue
            try:
                live["latest"] = inspect(frame, "live", f"camera{index}_{datetime.now():%H%M%S}")["id"]
                live["frames"] += 1
                live["error"] = None
            except Exception as e:
                live["error"] = str(e)
            time.sleep(args.every)

    if args.watch:
        threading.Thread(target=watch_loop, args=(args.watch,), daemon=True).start()
    elif args.camera is not None:
        threading.Thread(target=camera_loop, args=(args.camera,), daemon=True).start()

    # ---------- routes ----------
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"))

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/config")
    def config():
        return jsonify({
            "model": predictor.name, "demo": predictor.demo,
            "model_threshold": predictor.threshold, "settings": settings,
            "positions": [{"name": n, "x": float(u), "y": float(v)} for n, (u, v) in zip(pipe.names, pipe.pts)],
            "image_size": [640, 480], "crop_px": pipe.crop_px,
            "samples": len(samples), "live": live["source"],
        })

    @app.post("/api/settings")
    def set_settings():
        body = request.get_json(force=True)
        if "threshold" in body:
            t = body["threshold"]
            settings["threshold"] = predictor.threshold if t is None else min(1.0, max(0.0, float(t)))
        if "fill_count" in body:
            fc = body["fill_count"]
            settings["fill_count"] = None if fc in (None, "", "auto") else min(30, max(0, int(fc)))
        return jsonify(settings)

    @app.post("/api/inspect")
    def inspect_upload():
        f = request.files.get("image")
        if not f:
            abort(400, "send the photo as form field 'image'")
        return jsonify(view(inspect(decode(f.read()), "upload", f.filename or "upload")))

    @app.get("/api/result/<rid>")
    def result(rid):
        r = store.get(rid) or abort(404)
        thr = request.args.get("threshold", type=float)
        fill = request.args.get("fill")
        fill = None if fill in (None, "", "auto") else int(fill)
        if thr is None and fill is None:
            return jsonify(view(r))
        return jsonify(view(r, thr if thr is not None else settings["threshold"], fill))

    @app.get("/api/image/<rid>")
    def image(rid):
        path = os.path.join(store.out, "images", os.path.basename(rid) + ".jpg")
        return send_file(path, mimetype="image/jpeg") if os.path.exists(path) else abort(404)

    @app.get("/api/annotated/<rid>")
    def annotated(rid):
        r = store.get(rid) or abort(404)
        bgr = cv2.imread(os.path.join(store.out, "images", rid + ".jpg"))
        ok, buf = cv2.imencode(".jpg", pipe.annotate(bgr, r))
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg",
                         download_name=f"{rid}_{r['verdict']}.jpg", as_attachment=True)

    # ---------- order tracking ----------
    def with_findings(summary):
        summary["findings"] = findings.from_order(summary)
        return summary

    @app.get("/api/order")
    def order_state():
        return jsonify(with_findings(tracker.summary()))

    @app.post("/api/order/start")
    def order_start():
        """Start an order; the photo with this inspection id (default: the latest) is the empty tray."""
        body = request.get_json(silent=True) or {}
        rid = body.get("id") or (store.order[-1] if store.order else None)
        if not rid or not store.get(rid):
            abort(400, "inspect a photo of the empty tray first, then start the order")
        if store.get(rid)["verdict"] == "SKIPPED":
            abort(400, "that photo has the robot arm over the fixture; use one with the arm clear")
        bgr = cv2.imread(os.path.join(store.out, "images", rid + ".jpg"))
        with run_lock:
            return jsonify(with_findings(tracker.start(bgr, store.get(rid)["name"])))

    @app.post("/api/order/placed")
    def order_placed():
        with run_lock:
            return jsonify(with_findings(tracker.robot_placed()))

    @app.post("/api/order/complete")
    def order_complete():
        body = request.get_json(silent=True) or {}
        with run_lock:
            return jsonify(with_findings(tracker.order_complete(body.get("count"))))

    @app.post("/api/order/stop")
    def order_stop():
        with run_lock:
            tracker.reset()
            return jsonify(tracker.summary())

    @app.get("/api/order/report")
    def order_report():
        """The order inspection record as one HTML page (also saved in <out>/reports/)."""
        if not tracker.active:
            abort(400, "no order is being tracked; start one on a photo of the empty tray")
        with run_lock:
            page = report.build(tracker, findings.from_order(tracker.summary()), {"model": predictor.name})
        os.makedirs(os.path.join(store.out, "reports"), exist_ok=True)
        stem = os.path.splitext(os.path.basename(str(tracker.ref_name)))[0]
        path = os.path.join(store.out, "reports", f"order_{stem}_{datetime.now():%H%M%S}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        return Response(page, mimetype="text/html")

    @app.get("/api/order/record.csv")
    def order_csv():
        if not tracker.active:
            abort(400, "no order is being tracked")
        s = tracker.summary()
        gaps = {r["position"]: r for r in s["cycle"]["placements"]}
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["position", "status", "placed_at", "gap_s", "slow", "change", "cap_colour_at_centre",
                    "model_cap_defect_score", "photo"])
        for p in s["positions"]:
            g = gaps.get(p["position"], {})
            w.writerow([p["position"], p["state"],
                        datetime.fromtimestamp(p["placed_at"]).isoformat(timespec="seconds") if p.get("placed_at") else "",
                        g.get("gap_s", ""), g.get("slow", ""), p.get("change", ""), p.get("cap_colour_at_centre", ""),
                        p.get("cap_defect_score", ""), p.get("frame", "")])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=order_record.csv"})

    @app.get("/api/history")
    def history():
        return jsonify(store.summaries())

    @app.get("/api/history.csv")
    def history_csv():
        return Response(open(store.history_path).read(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=inspection_history.csv"})

    @app.post("/api/samples/<int:i>")
    def sample(i):
        if not samples:
            abort(404, "start the server with --samples <folder>")
        i %= len(samples)
        bgr = cv2.imread(os.path.join(args.samples, samples[i]))
        if bgr is None:   # a corrupt or half-written file: skip it, don't stop the replay
            return jsonify({"skipped": True, "reason": "could not read this image", "name": samples[i],
                            "sample_index": i, "sample_total": len(samples)})
        try:
            out = view(inspect(bgr, "sample", samples[i]))
        except Exception as e:  # one bad photo must not end a demo; log it and move on
            app.logger.exception("inspection failed for %s", samples[i])
            return jsonify({"skipped": True, "reason": f"inspection failed: {e}", "name": samples[i],
                            "sample_index": i, "sample_total": len(samples)})
        out["sample_index"], out["sample_total"] = i, len(samples)
        return jsonify(out)

    @app.get("/api/live")
    def live_state():
        r = store.get(live["latest"]) if live["latest"] else None
        return jsonify({"source": live["source"], "frames": live["frames"], "error": live["error"],
                        "latest": view(r) if r else None})

    @app.errorhandler(400)
    @app.errorhandler(404)
    def err(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="trained checkpoint, e.g. runs/twin_v4/best.pt")
    ap.add_argument("--demo", action="store_true", help="no model: fake results from a colour rule (for UI work)")
    ap.add_argument("--order-only", action="store_true",
                    help="order tracking without the cap classifier (no torch or model needed; faster)")
    ap.add_argument("--threshold", type=float, default=None, help="override the checkpoint's threshold")
    ap.add_argument("--samples", help="folder of photos to step through in the app")
    live = ap.add_mutually_exclusive_group()
    live.add_argument("--watch", help="folder to watch for new photos")
    live.add_argument("--camera", type=int, help="camera index (cv2.VideoCapture)")
    ap.add_argument("--include-existing", action="store_true", help="with --watch, also inspect photos already there")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between live checks")
    ap.add_argument("--no-align", action="store_true", help="always use the calibrated crop positions")
    ap.add_argument("--out", default="app_runs")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to let other computers on the network connect")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not (args.demo or args.order_only or args.model):
        ap.error("give --model path/to/best.pt, or --order-only (order tracking without the model)")
    app = build_app(args)
    print(f"open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

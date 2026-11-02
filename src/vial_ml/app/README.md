# Vial inspection app

A local web app that turns the trained model into an inspection station screen: load a photo of the tray, get PASS / REJECT / CHECK, see which positions are bad and why.

## Run it

From the `vial_ml` folder:

```bash
pip install flask opencv-python numpy torch torchvision pillow

python app/server.py --model runs/twin_v4/best.pt --samples path/to/cap_seal/images
```

Open http://localhost:8000 in a browser.

| Want to | Command |
| --- | --- |
| Upload photos in the browser | `python app/server.py --model best.pt` |
| Step through a folder (←/→ keys) | add `--samples <folder>` |
| Live: inspect new photos as the station saves them | add `--watch <folder>` |
| Live: a camera plugged into this computer | add `--camera 0` |
| **Demo: order tracking only** (no cap classifier, faster, no torch needed) | `python app/server.py --order-only --samples <folder>` |
| Try the screen without torch or a model | `python app/server.py --demo --samples <folder>` (fake results, clearly marked DEMO) |
| Let teammates on the same network open it | add `--host 0.0.0.0`, then they open `http://<your-ip>:8000` |

Every inspection is saved in `app_runs/`: the photo, the full result as JSON, and a line in `history.csv` (also downloadable from the History panel).

## What happens to each photo

1. **Image processing.** Resized to 640×480 if needed. Skipped if the robot arm (a large yellow area) covers the fixture.
2. **Image checks.** Brightness and sharpness of the 30 crop windows are compared with the real station camera (brightness 145–177, sharpness 2700+ on the 89 cap_seal frames). Outside the limits the result becomes CHECK rather than PASS.
3. **Fixture cropping and alignment.** Crops are cut at the 30 calibrated G positions (`real_positions.json`). The app measures where the yellow caps actually are; if the whole fixture has moved 6–15 px, the crops follow it. Over 15 px it warns instead. On all real frames the measured offset was under 2 px, so normally nothing moves.
4. **Model.** Each 64 px crop → probabilities for pass, missing_cap, crooked_cap, damaged_vial, no_vial.
5. **Catch-all rule.** Defect score = P(missing)+P(crooked)+P(damaged). Flagged if ≥ the threshold saved in the checkpoint (chosen on real validation data). The slider changes it live.
6. **Fill pattern.** The line fills G01, G02, … in order. The fill count is the highest occupied position (or set it by hand). An empty hole inside the fill count is a **missed placement**. With a manual count, a vial past it is marked for checking.
7. **Verdict.** REJECT if any defect or missed placement; CHECK for image warnings or vials past the fill count; otherwise PASS.

## Order tracking (placement by placement)

The station's robot fills the good fixture one vial at a time, in a fixed order (G01–G05 is the rightmost column, top to bottom, then the next column left). Order tracking uses that instead of judging every photo from scratch.

1. Show a photo of the **empty tray** and press **Start order**. That photo is the reference for all 30 holes.
2. Every later photo (with the arm clear) is compared with the reference, hole by hole. Because each hole is compared with itself, lighting, plate colour and glare cancel out.

| Change from the empty tray | Meaning |
| --- | --- |
| under 20 | still empty |
| 20–55, and no cap colour at the centre | a vial with an **open mouth** (missing cap) |
| 55 or more | a capped vial; the model checks it for a crooked cap or damage |

3. Events:
   - **placed OK**;
   - **open mouth**;
   - **flagged by model**;
   - **skipped**: a later position filled while this one stays empty;
   - **missed pick**: after **Robot placed a vial**, nothing appears;
   - **changed after inspection**: a cap moved or a vial was knocked;
   - **vial gone**;
   - **missing at completion**: after **Order complete**.
4. A hole is only judged once it looks the same in two photos in a row, so a passing arm or a vial still settling doesn't trigger anything.

**How it was checked.** The thresholds come from the real validation positions only: empty holes change at most 11.7, open vials 37–43, capped vials 64–86. Replaying the three real orders that start with an empty tray matches the labels exactly: 25 good placements accepted, the 2 open vials (G17, G20) caught. Skipped holes, missed picks, removed and knocked vials were tested by editing real photos (`tests/test_order.py`).

**Limits.**
- Only 7 open-vial examples, all from one order.
- No real crooked or damaged vials in the clean sequences, so those still rely on the model.
- The open-mouth check uses the cap colour (yellow), so a new cap colour needs that setting changed.
- G01, a test position, was looked at while building the cap-colour check. New station photos are the clean final test.

API: `POST /api/order/start {id}`, `POST /api/order/placed`, `POST /api/order/complete {count}`, `POST /api/order/stop`, `GET /api/order`. Each inspection result also carries the order state in `result["order"]`.

## Order record and cycle times

While an order is tracked, **Order record** (in the Order tracking panel) opens a one-page inspection record you can print or save as PDF. A copy is saved in `app_runs/reports/`. It holds:
- verdict, order times and the organizers' tray findings;
- every problem with photo evidence: the empty-tray crop next to the crop at the moment it happened, with the time;
- the fixture with a photo of each vial as it was placed;
- cycle time between placements, with unusually slow ones marked (over twice the median, and at least a minute over it);
- all 30 positions and the full event log.

Placement time is the capture time of the first photo a vial appears in, read from the station's file name (`cap_seal_YYYYMMDD_HHMMSS_…`), otherwise from when the file was written. `GET /api/order/record.csv` gives the same table as CSV.

## For the backend

`app/inspection.py` has no web or torch code in it:

```python
from inspection import Pipeline, ModelPredictor
pipe = Pipeline(ModelPredictor("best.pt"))
result = pipe.run(bgr_image)            # verdict, positions, fill, quality, alignment
```

HTTP endpoints (all JSON): `POST /api/inspect` (form field `image`), `GET /api/result/<id>?threshold=&fill=`, `POST /api/settings`, `GET /api/history`, `GET /api/history.csv`, `GET /api/live`, `GET /api/annotated/<id>` (marked-up JPEG).

## Files

| File | What it does |
| --- | --- |
| `server.py` | web server, live folder/camera sources, saving results |
| `inspection.py` | image checks, alignment, cropping, model call, fill pattern, verdict |
| `order.py` | order tracking: empty-tray reference, placement events |
| `static/index.html` | the screen (no build step, no external libraries) |
| `../tests/test_app.py` | tests for the verdict and fill-pattern rules |

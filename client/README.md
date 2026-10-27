# VialSight image upload demo

The client is a Bun + React interface for exercising the repository's `/inspect` upload endpoint.

## Run locally

```bash
bun install
bun run dev
```

Open the URL printed by Bun. To create a production build, run `bun run build`.

## Current behavior

- Accepts one image up to 20 MB, with file-picker and drag-and-drop input.
- Shows a local preview before sending the image as multipart form data to `POST /inspect`.
- Displays the endpoint's response and request errors.
- Does not classify the image. The endpoint currently only acknowledges receipt; it does not connect to a trained model or return inspection predictions.

The image is sent to the local demo endpoint, which does not retain or analyze it. The optional video control and placeholder live metrics have been removed because the endpoint does not support them.

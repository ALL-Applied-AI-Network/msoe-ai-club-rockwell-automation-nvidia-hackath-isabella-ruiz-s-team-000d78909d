import { serve } from "bun";
import index from "./index.html";

const server = serve({
  routes: {
    // Serve index.html for all unmatched routes.
    "/*": index,

    "/inspect": {
      async POST(req) {
        let formData: FormData;
        try {
          formData = await req.formData();
        } catch {
          return Response.json(
            { status: "error", message: "Request must contain valid multipart form data." },
            { status: 400 },
          );
        }

        const file = formData.get("file");
        if (!(file instanceof File)) {
          return Response.json(
            { status: "error", message: 'A file is required in the "file" field.' },
            { status: 400 },
          );
        }

        if (!file.type.startsWith("image/")) {
          return Response.json(
            { status: "error", message: "The uploaded file must be an image." },
            { status: 400 },
          );
        }

        if (file.size === 0) {
          return Response.json(
            { status: "error", message: "The uploaded image is empty." },
            { status: 400 },
          );
        }

        return Response.json({
          filename: file.name,
          status: "received",
          message: "Image received. No AI inspection model is connected in this demo.",
          predictions: [],
        });
      },
    },

    "/api/hello": {
      async GET(req) {
        return Response.json({
          message: "Hello, world!",
          method: "GET",
        });
      },
      async PUT(req) {
        return Response.json({
          message: "Hello, world!",
          method: "PUT",
        });
      },
    },

    "/api/hello/:name": async req => {
      const name = req.params.name;
      return Response.json({
        message: `Hello, ${name}!`,
      });
    },
  },

  development: process.env.NODE_ENV !== "production" && {
    // Enable browser hot reloading in development
    hmr: true,

    // Echo console logs from the browser to the server
    console: true,
  },
});

console.log(`🚀 Server running at ${server.url}`);

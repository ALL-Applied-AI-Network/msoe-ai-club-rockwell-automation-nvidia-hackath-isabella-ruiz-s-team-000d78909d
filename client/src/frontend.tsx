import { StrictMode, useEffect, useState, type ChangeEvent, type ReactNode } from "react";
import { createRoot } from "react-dom/client";

type InspectionStatus = {
  kind: "success" | "error";
  message: string;
};

function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return url;
}

export async function submitInspection(file: File): Promise<unknown> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/inspect", { method: "POST", body: formData });
  const responseText = await response.text();
  let payload: unknown;

  try {
    payload = JSON.parse(responseText);
  } catch {
    throw new Error("The inspection service returned an invalid response.");
  }

  if (!response.ok) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "message" in payload &&
      typeof payload.message === "string"
        ? payload.message
        : `Inspection failed with status ${response.status}.`;
    throw new Error(message);
  }

  return payload;
}

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const shared = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };

  if (name === "vial") {
    return (
      <svg {...shared}>
        <path d="M9 3h6M10 3v5.1L5.8 15a4.5 4.5 0 0 0 3.9 6.7h4.6a4.5 4.5 0 0 0 3.9-6.7L14 8.1V3" />
        <path d="M8 15h8M9.5 18h5" />
      </svg>
    );
  }

  if (name === "image") {
    return (
      <svg {...shared}>
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <path d="m21 15-5-5L5 21" />
      </svg>
    );
  }

  if (name === "video") {
    return (
      <svg {...shared}>
        <rect x="3" y="5" width="13" height="14" rx="2" />
        <path d="m16 10 5-3v10l-5-3z" />
      </svg>
    );
  }

  if (name === "upload") {
    return (
      <svg {...shared}>
        <path d="M12 16V4m-4 4 4-4 4 4M4 16v4h16v-4" />
      </svg>
    );
  }

  if (name === "check") {
    return (
      <svg {...shared}>
        <path d="m5 12 4 4L19 6" />
      </svg>
    );
  }

  if (name === "close") {
    return (
      <svg {...shared}>
        <path d="m18 6-12 12M6 6l12 12" />
      </svg>
    );
  }

  return (
    <svg {...shared}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v4m0 4h.01" />
    </svg>
  );
}

function FileCard({
  kind,
  file,
  previewUrl,
  onSelect,
  onRemove,
}: {
  kind: "image" | "video";
  file: File | null;
  previewUrl: string | null;
  onSelect: (event: ChangeEvent<HTMLInputElement>) => void;
  onRemove: () => void;
}) {
  const isImage = kind === "image";
  const title = isImage ? "Image sample" : "Video sample";
  const accept = isImage ? "image/*" : "video/*";

  return (
    <section className="upload-card">
      <div className="card-heading">
        <div className={`card-icon ${isImage ? "image-icon" : "video-icon"}`}>
          <Icon name={kind} />
        </div>
        <div>
          <h2>{title}</h2>
          <p>{isImage ? "Single frame analysis" : "Optional process footage"}</p>
        </div>
        <span className="step-label">{isImage ? "REQUIRED" : "OPTIONAL"}</span>
      </div>

      <input
        accept={accept}
        className="visually-hidden"
        id={`${kind}-file`}
        onChange={onSelect}
        type="file"
      />

      {file && previewUrl ? (
        <div className="preview-area">
          {isImage ? (
            <img src={previewUrl} alt={`Preview of ${file.name}`} />
          ) : (
            <video src={previewUrl} controls aria-label={`Preview of ${file.name}`} />
          )}
          <div className="file-details">
            <div className="file-name-wrap">
              <span className="file-check"><Icon name="check" size={14} /></span>
              <div>
                <strong title={file.name}>{file.name}</strong>
                <span>{(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to inspect</span>
              </div>
            </div>
            <button
              className="icon-button"
              onClick={onRemove}
              type="button"
              aria-label={`Remove ${kind} file`}
              title="Remove file"
            >
              <Icon name="close" size={16} />
            </button>
          </div>
        </div>
      ) : (
        <label className="dropzone" htmlFor={`${kind}-file`}>
          <span className="upload-symbol"><Icon name="upload" size={22} /></span>
          <strong>{isImage ? "Choose an image" : "Choose a video"}</strong>
          <span>or <span className="browse-link">browse files</span></span>
          <span className="file-hint">
            {isImage ? "JPG, PNG, WEBP · Up to 20 MB" : "MP4, MOV, WEBM · Up to 100 MB"}
          </span>
        </label>
      )}

      <label className="change-file" htmlFor={`${kind}-file`}>
        {file ? "Choose a different file" : `Select ${isImage ? "image" : "video"}`}
        <span aria-hidden="true">↗</span>
      </label>
    </section>
  );
}

function Dashboard() {
  const [image, setImage] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [status, setStatus] = useState<InspectionStatus | null>(null);
  const [result, setResult] = useState<unknown>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const imageUrl = useObjectUrl(image);
  const videoUrl = useObjectUrl(video);

  const selectFile = (
    event: ChangeEvent<HTMLInputElement>,
    kind: "image" | "video",
  ) => {
    const file = event.currentTarget.files?.[0] ?? null;
    event.currentTarget.value = "";

    if (!file) return;

    const isImage = kind === "image";
    const validType = isImage ? file.type.startsWith("image/") : file.type.startsWith("video/");
    const maxSize = isImage ? 20 * 1024 * 1024 : 100 * 1024 * 1024;
    const label = isImage ? "image" : "video";

    setResult(null);
    if (!validType || file.size > maxSize) {
      if (isImage) setImage(null);
      else setVideo(null);
      setStatus({
        kind: "error",
        message: !validType
          ? `Please choose a valid ${label} file.`
          : `The ${label} file must be smaller than ${isImage ? "20" : "100"} MB.`,
      });
      return;
    }

    if (isImage) setImage(file);
    else setVideo(file);
    setStatus(null);
  };

  const removeFile = (kind: "image" | "video") => {
    if (kind === "image") setImage(null);
    else setVideo(null);
    setResult(null);
    setStatus(null);
  };

  const runInspection = async () => {
    if (!image) {
      setStatus({ kind: "error", message: "Select an image sample before running inspection." });
      return;
    }

    setIsSubmitting(true);
    setStatus(null);
    setResult(null);
    try {
      const response = await submitInspection(image);
      setResult(response);
      setStatus({ kind: "success", message: "Upload received. This demo does not run an AI inspection." });
    } catch (error) {
      setStatus({
        kind: "error",
        message: error instanceof Error ? error.message : "Inspection failed unexpectedly.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <style>{styles}</style>
      <div className="app-shell">
        <header className="topbar">
          <a className="brand" href="/" aria-label="VialSight home">
            <span className="brand-mark"><Icon name="vial" size={22} /></span>
            <span>Vial<span className="brand-light">Sight</span></span>
            <span className="brand-divider" />
            <span className="brand-caption">QUALITY INTELLIGENCE</span>
          </a>
          <div className="system-status"><span className="online-dot" /> Demo workspace</div>
        </header>

        <main>
          <section className="intro">
            <div>
              <div className="eyebrow"><span /> INSPECTION WORKSPACE <span className="eyebrow-line" /></div>
              <h1>See the details.<br /><span>Trust the quality.</span></h1>
              <p className="intro-copy">
                Analyze vial samples with precision. Upload an image to start your inspection.
              </p>
            </div>
            <div className="session-chip"><span className="session-dot" /> DEMO WORKSPACE <span className="chip-divider" /> READY FOR UPLOAD</div>
          </section>

          <section className="metrics" aria-label="Inspection summary">
            <div className="metric">
              <span className="metric-label">INSPECTIONS TODAY</span>
              <strong>—</strong>
              <span className="metric-note">No live data</span>
            </div>
            <div className="metric">
              <span className="metric-label">PASS RATE</span>
              <strong>—</strong>
              <span className="metric-note">No live data</span>
            </div>
            <div className="metric">
              <span className="metric-label">AVG. PROCESSING</span>
              <strong>—</strong>
              <span className="metric-note">No live data</span>
            </div>
            <div className="metric metric-last">
              <span className="metric-label">MODEL STATUS</span>
              <strong className="model-ready">—</strong>
              <span className="metric-note">No AI model connected</span>
            </div>
          </section>

          <section className="workspace-heading">
            <div>
              <h2>New inspection</h2>
              <p>Provide a sample to run a quality check.</p>
            </div>
            <span className="required-note"><span>*</span> Required field</span>
          </section>

          <div className="upload-grid">
            <FileCard
              kind="image"
              file={image}
              previewUrl={imageUrl}
              onSelect={(event) => selectFile(event, "image")}
              onRemove={() => removeFile("image")}
            />
            <FileCard
              kind="video"
              file={video}
              previewUrl={videoUrl}
              onSelect={(event) => selectFile(event, "video")}
              onRemove={() => removeFile("video")}
            />
          </div>

          <section className="action-row">
            <div className="privacy-note"><span className="privacy-icon">◈</span> Your samples are processed securely and never stored.</div>
            <button className="run-button" disabled={isSubmitting} onClick={runInspection} type="button">
              {isSubmitting ? "Inspecting sample…" : <>Run inspection <span>→</span></>}
            </button>
          </section>

          <section className="results-panel">
            <div className="results-head">
              <div className="results-title"><span className="result-icon"><Icon name="check" size={16} /></span><div><h2>Inspection results</h2><p>Analysis output will appear here</p></div></div>
              <span className="output-label">OUTPUT</span>
            </div>
            {status ? (
              <div className={`status-message ${status.kind}`} role="status">
                <span className="status-indicator"><Icon name={status.kind === "success" ? "check" : "alert"} size={16} /></span>
                {status.message}
              </div>
            ) : (
              <div className="empty-result"><span className="empty-result-mark">{"{ }"}</span>Waiting for an inspection to run<span className="cursor">_</span></div>
            )}
            {result !== null && <pre className="json-result">{JSON.stringify(result, null, 2)}</pre>}
          </section>
        </main>

      </div>
    </>
  );
}

const styles = `
  :root{font-family:'DM Sans',sans-serif;color:#e6edee;background:#0a1012;font-synthesis:none;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;font-optical-sizing:auto;font-weight:400;--muted:#778588;--line:#202b2d;--teal:#5be0c0}
  *{box-sizing:border-box}
  body{margin:0;min-width:320px;min-height:100vh;background:radial-gradient(ellipse at 78% 8%,rgba(28,76,69,.18),transparent 38%),#0a1012}
  button,input{font:inherit}
  button{cursor:pointer}
  .app-shell{max-width:1440px;min-height:100vh;margin:0 auto;padding:0 7.1%}
  .topbar{height:76px;border-bottom:1px solid rgba(125,155,157,.15);display:flex;align-items:center;justify-content:space-between}
  .brand{display:flex;align-items:center;gap:10px;color:#ecf4f4;text-decoration:none;font-family:Manrope,sans-serif;font-size:19px;font-weight:800;letter-spacing:-.8px}
  .brand-mark{width:32px;height:32px;border:1px solid rgba(91,224,192,.32);background:rgba(91,224,192,.08);color:var(--teal);display:grid;place-items:center;border-radius:9px}
  .brand-light{font-weight:500;color:#a8bbba}.brand-divider{height:19px;width:1px;background:#293638;margin:0 4px 0 8px}.brand-caption,.system-status,.eyebrow,.session-chip,.metric-label,.required-note,.step-label,.output-label,.footer-right{font-family:'DM Mono',monospace;letter-spacing:.07em}
  .brand-caption{font-size:9px;letter-spacing:.12em;color:#748486;font-weight:400}
  .system-status{display:flex;align-items:center;gap:9px;color:#99aaa9;font-size:10px;letter-spacing:.02em}
  .online-dot{width:7px;height:7px;display:inline-block;background:#55d9a9;border-radius:50%;box-shadow:0 0 11px rgba(85,217,169,.48)}
  main{max-width:1070px;margin:0 auto}
  .intro{display:flex;justify-content:space-between;align-items:flex-end;padding:61px 0 34px}
  .eyebrow{display:flex;align-items:center;gap:9px;color:#88a09f;font-size:9px;letter-spacing:.15em;margin-bottom:18px}
  .eyebrow>span:first-child{width:6px;height:6px;border-radius:50%;background:var(--teal)}.eyebrow-line{width:37px;height:1px;background:#344447}
  h1,h2,p{margin:0}
  h1{font-family:Manrope,sans-serif;font-size:clamp(36px,4.4vw,54px);line-height:1.09;letter-spacing:-2.5px;font-weight:600;color:#eff5f3}
  h1 span{color:#67cbb5}
  .intro-copy{color:#879594;font-size:13px;margin-top:15px;letter-spacing:.01em}
  .session-chip{display:flex;align-items:center;gap:8px;border:1px solid #263637;background:rgba(16,28,29,.7);border-radius:5px;padding:9px 11px;color:#8da09e;font-size:8px;letter-spacing:.08em;margin-bottom:5px}
  .session-dot{width:6px;height:6px;border-radius:50%;background:#55d9a9}.chip-divider{height:12px;width:1px;background:#314143;margin:0 2px}
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid #253133;border-radius:7px;background:linear-gradient(110deg,rgba(19,30,31,.95),rgba(15,24,25,.9));padding:19px 0;margin-bottom:44px}
  .metric{min-height:77px;padding:0 25px;border-right:1px solid #273436;display:flex;flex-direction:column;align-items:flex-start;gap:7px}
  .metric-last{border:0}.metric-label{font-size:8px;color:#849393;letter-spacing:.1em}.metric strong{font-family:Manrope,sans-serif;font-size:23px;line-height:1.1;font-weight:600;letter-spacing:-.8px;color:#e7efed}.metric-unit{font-size:14px;color:#a5b5b2;margin-left:2px}.metric-note{font-size:10px;color:#748482}.metric-note.positive{color:#63d0ad}.metric-note span{color:#748482}.model-ready{font-size:16px!important;display:flex;align-items:center;gap:8px}
  .workspace-heading{display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:16px}.workspace-heading h2,.results-title h2{font-family:Manrope,sans-serif;font-size:16px;font-weight:600;letter-spacing:-.25px;color:#e5ecea}.workspace-heading p,.results-title p{font-size:11px;color:#82908f;margin-top:5px}.required-note{font-size:9px;color:#81908e;letter-spacing:.02em}.required-note span{color:#5be0c0}
  .upload-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .upload-card{padding:18px 19px 13px;border:1px solid #263436;background:rgba(17,26,28,.88);border-radius:7px;min-width:0}
  .card-heading{display:flex;align-items:center;gap:11px;margin-bottom:16px}.card-icon{width:34px;height:34px;border-radius:7px;display:grid;place-items:center}.image-icon{background:rgba(53,184,155,.12);color:#68dbbd}.video-icon{background:rgba(112,148,203,.13);color:#87a9dc}.card-heading h2{font-size:12px;font-weight:600;color:#e3ebea}.card-heading p{font-size:10px;color:#7e8e8d;margin-top:4px}.step-label{margin-left:auto;color:#738583;border:1px solid #2c3a3b;padding:4px 6px;font-size:7px;border-radius:3px;letter-spacing:.08em}
  .dropzone{height:177px;border:1px dashed #354446;border-radius:5px;background:rgba(9,15,17,.42);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:7px;color:#a2b0ae;font-size:11px;cursor:pointer;transition:border-color .18s,background .18s}
  .dropzone:hover{border-color:#58cdb0;background:rgba(41,104,89,.08)}.upload-symbol{color:#6acbb4;margin-bottom:2px}.dropzone strong{font-size:11px;font-weight:500;color:#cad4d2}.browse-link{color:#67d9bd;text-decoration:underline;text-underline-offset:3px}.file-hint{font-family:'DM Mono',monospace;font-size:8px;color:#718180;margin-top:4px;letter-spacing:.025em}
  .change-file{height:29px;display:flex;justify-content:space-between;align-items:flex-end;color:#9eadaa;font-size:9px;cursor:pointer}.change-file span{color:#6c8581;font-size:13px}.change-file:hover{color:#65d7bb}
  .preview-area{height:177px;position:relative;overflow:hidden;border-radius:5px;background:#080d0f;border:1px solid #263436;display:flex;align-items:center;justify-content:center}
  .preview-area>img,.preview-area>video{width:100%;height:100%;object-fit:contain;background:#080d0f}.preview-area>video{object-fit:contain}
  .file-details{position:absolute;bottom:0;left:0;right:0;padding:9px 10px;background:linear-gradient(0deg,rgba(7,13,14,.97),rgba(7,13,14,.8));display:flex;align-items:center;justify-content:space-between;gap:8px}
  .file-name-wrap{display:flex;align-items:center;gap:8px;min-width:0}.file-check{width:20px;height:20px;border-radius:50%;background:rgba(85,217,169,.14);color:#60d5ae;display:grid;place-items:center;flex:none}.file-name-wrap>div{display:flex;flex-direction:column;gap:3px;min-width:0}.file-name-wrap strong{font-size:9px;color:#dce5e3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:250px}.file-name-wrap>div>span{font-size:8px;color:#84928f}.icon-button{width:26px;height:26px;display:grid;place-items:center;border:1px solid #3a4848;border-radius:4px;color:#9baaa7;background:rgba(10,17,18,.76);flex:none}.icon-button:hover{color:#fff;border-color:#71827f}
  .visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
  .visually-hidden:focus-visible+label{outline:2px solid #5be0c0;outline-offset:3px}
  .action-row{display:flex;align-items:center;justify-content:space-between;margin:17px 0 26px}.privacy-note{font-size:9px;color:#81908e;display:flex;align-items:center;gap:8px}.privacy-icon{font-size:15px;color:#78aa9e}
  .run-button{height:42px;border:0;border-radius:5px;background:#5bd5b5;color:#09201b;padding:0 19px;font-size:11px;font-weight:700;letter-spacing:.01em;box-shadow:0 4px 18px rgba(62,199,165,.1);transition:background .2s,transform .2s}.run-button span{font-size:16px;margin-left:12px}.run-button:hover:not(:disabled){background:#78e6c9;transform:translateY(-1px)}.run-button:disabled{opacity:.6;cursor:wait}
  .results-panel{border:1px solid #263436;background:rgba(15,24,25,.88);border-radius:7px;min-height:156px;padding:16px 18px}
  .results-head{display:flex;align-items:center;justify-content:space-between;padding-bottom:14px;border-bottom:1px solid #253133}.results-title{display:flex;align-items:center;gap:10px}.result-icon{width:27px;height:27px;border:1px solid #334345;border-radius:5px;color:#83a59e;display:grid;place-items:center}.results-title h2{font-size:12px}.results-title p{font-size:9px;margin-top:3px}.output-label{font-size:8px;color:#788985;letter-spacing:.13em}
  .empty-result{font-family:'DM Mono',monospace;color:#72817f;font-size:10px;min-height:70px;display:flex;align-items:center;gap:8px}.empty-result-mark{font-size:14px;color:#527a70}.cursor{color:#65d9b9;animation:blink 1.1s step-end infinite}.status-message{display:flex;align-items:center;gap:9px;padding:15px 0 8px;font-size:11px}.status-message.success{color:#78dbb8}.status-message.error{color:#ff9d8e}.status-indicator{display:grid;place-items:center;width:22px;height:22px;border-radius:50%;background:rgba(91,224,192,.1)}.status-message.error .status-indicator{background:rgba(255,130,112,.1)}.json-result{max-height:280px;overflow:auto;margin:7px 0 0;padding:13px;border:1px solid #243234;border-radius:4px;background:#0b1214;color:#a8d9cb;font:10px/1.6 'DM Mono',monospace;white-space:pre-wrap;overflow-wrap:anywhere}
  footer{max-width:1070px;margin:33px auto 0;border-top:1px solid rgba(125,155,157,.14);padding:16px 0 19px;display:flex;justify-content:space-between;color:#687775;font-size:9px}
  @keyframes blink{50%{opacity:0}}
  @media(max-width:720px){.app-shell{padding:0 5%}.topbar{height:65px}.brand{font-size:17px}.brand-caption{display:none}.brand-divider{display:none}.system-status{font-size:0}.system-status .online-dot{width:8px;height:8px}.intro{padding:42px 0 26px;display:block}.intro h1{font-size:43px}.session-chip{display:inline-flex;margin-top:22px}.metrics{grid-template-columns:1fr 1fr;margin-bottom:34px;padding:13px 0}.metric{padding:12px 16px;min-height:78px}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid #273436}.upload-grid{grid-template-columns:1fr;gap:12px}.workspace-heading{margin-bottom:13px}.action-row{align-items:flex-start;flex-direction:column;gap:15px}.run-button{width:100%}.results-panel{min-height:145px}footer{margin-top:26px;gap:12px;flex-wrap:wrap}.footer-right{margin-left:auto}}
  @media(max-width:390px){.app-shell{padding:0 17px}.intro h1{font-size:38px}.session-chip{font-size:7px;padding:8px}.metric{padding:11px 12px}.metric-label{font-size:7px}.upload-card{padding:15px}.privacy-note{font-size:8px}}
  @media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important}}
`;

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Missing #root element for the VialSight app.");

const app: ReactNode = (
  <StrictMode>
    <Dashboard />
  </StrictMode>
);

(import.meta.hot.data.root ??= createRoot(rootElement)).render(app);

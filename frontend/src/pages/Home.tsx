import { useEffect, useMemo, useRef, useState } from "react";
import JSZip from "jszip";

type RefItem = {
  id: string;
  name: string;
  src: string;
  kind: "folder" | "generated" | "uploaded";
};

type MVResponse = {
  job_id: string;
  object_url?: string;
  reference_url?: string;
  obj_frames?: string[];
  pred_frames?: string[];
  frames?: string[];
};

type Vec3 = [number, number, number];

type OrbitAxis = "x" | "y" | "z";

type Quat = [number, number, number, number];

type OrbitSpec = {
  tiltDeg: number;
  headingDeg: number;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  quat: Quat;
  axis: Vec3;
};

function mod1(x: number) {
  const m = x % 1;
  return m < 0 ? m + 1 : m;
}

function filenameToName(s: string) {
  const base = s.replace(/\.[^/.]+$/, "");
  return base
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function pathToId(p: string) {
  return p.replace(/[^a-zA-Z0-9]+/g, "_");
}

const deg2rad = (d: number) => (d * Math.PI) / 180;

const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (n < 1e-8) return IDENTITY_QUAT;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function quatMul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return quatNormalize([
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ]);
}

function quatFromAxisAngle(axis: Vec3, deg: number): Quat {
  const n = norm3(axis);
  const h = deg2rad(deg) * 0.5;
  const s = Math.sin(h);
  return quatNormalize([n[0] * s, n[1] * s, n[2] * s, Math.cos(h)]);
}

function quatRotateVec(q: Quat, v: Vec3): Vec3 {
  const qv: Vec3 = [q[0], q[1], q[2]];
  const uv = cross(qv, v);
  const uuv = cross(qv, uv);
  return [
    v[0] + 2 * (q[3] * uv[0] + uuv[0]),
    v[1] + 2 * (q[3] * uv[1] + uuv[1]),
    v[2] + 2 * (q[3] * uv[2] + uuv[2]),
  ];
}

function quatToAxisAngle(q: Quat): { axis: Vec3; deg: number } {
  const n = quatNormalize(q);
  const w = Math.max(-1, Math.min(1, n[3]));
  const angle = 2 * Math.acos(w);
  const s = Math.sqrt(Math.max(0, 1 - w * w));

  if (s < 1e-6 || angle < 1e-6) {
    return { axis: [0, 1, 0], deg: 0 };
  }

  return {
    axis: [n[0] / s, n[1] / s, n[2] / s],
    deg: (angle * 180) / Math.PI,
  };
}

function orbitFromQuat(
  quat: Quat,
  yawDeg: number = 0,
  pitchDeg: number = 0,
  rollDeg: number = 0,
): OrbitSpec {
  const q = quatNormalize(quat);
  const axis = norm3(quatRotateVec(q, [0, 0, 1]));

  return {
    tiltDeg: 90,
    headingDeg: wrapDeg(yawDeg),
    yawDeg: wrapDeg(yawDeg),
    pitchDeg: wrapDeg(pitchDeg),
    rollDeg: wrapDeg(rollDeg),
    quat: q,
    axis,
  };
}

function orbitFromAngles(
  yawDeg: number,
  pitchDeg: number,
  rollDeg: number = 0,
): OrbitSpec {
  const qy = quatFromAxisAngle([0, 1, 0], yawDeg);
  const qx = quatFromAxisAngle([1, 0, 0], pitchDeg);
  const qz = quatFromAxisAngle([0, 0, 1], rollDeg);

  return orbitFromQuat(quatMul(quatMul(qy, qx), qz), yawDeg, pitchDeg, rollDeg);
}

function wrapDeg(d: number) {
  return ((d % 360) + 360) % 360;
}

function norm3(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n < 1e-8) return [0, -1, 0];
  return [v[0] / n, v[1] / n, v[2] / n];
}

function preloadFrame(src: string) {
  return new Promise<void>((resolve) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve();
    img.onerror = () => resolve();
    img.src = src;
  });
}

async function preloadFrames(srcs: string[]) {
  await Promise.all(srcs.map((src) => preloadFrame(src)));
  return srcs;
}

export default function Home() {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    if (typeof window === "undefined") return "dark";
    const saved = window.localStorage.getItem("theme");
    if (saved === "dark" || saved === "light") return saved;
    const prefersDark =
      window.matchMedia?.("(prefers-color-scheme: dark)")?.matches ?? true;
    return prefersDark ? "dark" : "light";
  });

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("theme", theme);
  }, [theme]);

  const folderRefs = useMemo<RefItem[]>(() => {
    const modules = import.meta.glob(
      "../assets/references/*.{png,jpg,jpeg,webp,avif}",
      { eager: true, import: "default" },
    ) as Record<string, string>;

    const items = Object.entries(modules).map(([path, url]) => {
      const file = path.split("/").pop() ?? path;
      return {
        id: `folder_${pathToId(path)}`,
        name: filenameToName(file),
        src: url,
        kind: "folder" as const,
      };
    });

    items.sort((a, b) => a.name.localeCompare(b.name));
    return items;
  }, []);

  const [objectImage, setObjectImage] = useState<string | null>(null);
  const [referenceImage, setReferenceImage] = useState<string | null>(null);
  const [activeRefId, setActiveRefId] = useState<string>("none");

  const [customRefs, setCustomRefs] = useState<RefItem[]>([]);
  const [refPrompt, setRefPrompt] = useState("");
  const [generatingRef, setGeneratingRef] = useState(false);

  const [objFrames, setObjFrames] = useState<string[]>([]);
  const [predFrames, setPredFrames] = useState<string[]>([]);

  const [yaw01, setYaw01] = useState(0);
  const [autoSpin, setAutoSpin] = useState(false);
  const [running, setRunning] = useState(false);

  const [orbitDraft, setOrbitDraft] = useState<OrbitSpec | null>(null);
  const [orbitConfirmed, setOrbitConfirmed] = useState<OrbitSpec | null>(null);

  const orbitDrag = useRef<{
    active: boolean;
    axis: OrbitAxis;
    lastX: number;
    lastY: number;
    centerX: number;
    centerY: number;
    quat: Quat;
  } | null>(null);

  const [activeOrbitAxis, setActiveOrbitAxis] = useState<OrbitAxis | null>(
    null,
  );

  const [isGizmoHovered, setIsGizmoHovered] = useState(false);

  const displayOrbit = useMemo(
    () => orbitDraft ?? orbitConfirmed ?? orbitFromAngles(0, 0, 0),
    [orbitDraft, orbitConfirmed],
  );

  const objInputRef = useRef<HTMLInputElement | null>(null);
  const refUploadInputRef = useRef<HTMLInputElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  const refStripRef = useRef<HTMLDivElement | null>(null);
  const [refScroll01, setRefScroll01] = useState(0);

  const [savingZip, setSavingZip] = useState(false);

  const dragRef = useRef<{
    active: boolean;
    startX: number;
    startYaw01: number;
    width: number;
    pointerId: number | null;
  } | null>(null);

  const refs = useMemo(
    () => [...customRefs, ...folderRefs],
    [customRefs, folderRefs],
  );

  const hasPredFrames = predFrames.length > 0;
  const hasObjFrames = objFrames.length > 0;

  const activeFrameKind =
    referenceImage && hasPredFrames ? "pred" : hasObjFrames ? "obj" : null;

  const activeFrames =
    activeFrameKind === "pred"
      ? predFrames
      : activeFrameKind === "obj"
        ? objFrames
        : [];

  const hasFrames = activeFrames.length > 0;

  const frameCount = activeFrames.length;
  const frameIndex = useMemo(() => {
    if (!hasFrames) return 0;
    const idx = Math.round(yaw01 * (frameCount - 1));
    return Math.max(0, Math.min(frameCount - 1, idx));
  }, [yaw01, hasFrames, frameCount]);

  const currentFrame = hasFrames ? activeFrames[frameIndex] : null;

  const frameLabel = useMemo(() => {
    if (!hasFrames || !activeFrameKind) return "";
    return `${activeFrameKind}_${String(frameIndex).padStart(3, "0")}.png`;
  }, [hasFrames, frameIndex, activeFrameKind]);

  const canRun = useMemo(
    () => Boolean(objectImage) && !running,
    [objectImage, running],
  );

  const resetOutputs = () => {
    setObjFrames([]);
    setPredFrames([]);
    setYaw01(0);
    setAutoSpin(false);
  };

  const clearPredictedFrames = () => {
    setPredFrames([]);
  };

  const setReference = (src: string | null, id: string) => {
    setReferenceImage(src);
    setActiveRefId(id);
    clearPredictedFrames();
  };

  const pickObject = () => objInputRef.current?.click();

  const onObjectFile = (file: File | null) => {
    if (!file) return;
    setObjectImage(URL.createObjectURL(file));
    resetOutputs();
  };

  const pickReferenceUpload = () => refUploadInputRef.current?.click();

  const onReferenceFile = (file: File | null) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const item: RefItem = {
      id: `upl_${crypto.randomUUID()}`,
      name: filenameToName(file.name || "Uploaded"),
      src: url,
      kind: "uploaded",
    };
    setCustomRefs((prev) => [item, ...prev]);
    setReference(item.src, item.id);
  };

  const beginDrag = (clientX: number, pointerId: number) => {
    if (!hasFrames) return;
    const el = viewportRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    dragRef.current = {
      active: true,
      startX: clientX,
      startYaw01: yaw01,
      width: Math.max(1, rect.width),
      pointerId,
    };
  };

  const moveDrag = (clientX: number) => {
    const st = dragRef.current;
    if (!st?.active) return;
    const delta = (clientX - st.startX) / st.width;
    setYaw01(mod1(st.startYaw01 + delta));
  };

  const endDrag = () => {
    const el = viewportRef.current;
    const st = dragRef.current;
    if (el && st?.active && st.pointerId != null) {
      try {
        el.releasePointerCapture(st.pointerId);
      } catch {
        // ignore
      }
    }
    if (dragRef.current) {
      dragRef.current.active = false;
      dragRef.current.pointerId = null;
    }
  };

  const stepFrame = (dir: 1 | -1) => {
    if (!hasFrames) return;
    const n = frameCount;
    const next = (frameIndex + dir + n) % n;
    setYaw01(n <= 1 ? 0 : next / (n - 1));
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!hasFrames) return;
      if (e.key === "ArrowLeft") stepFrame(-1);
      if (e.key === "ArrowRight") stepFrame(1);

      if (e.key === "Escape") {
        setOrbitDraft(null);
        setActiveOrbitAxis(null);
        if (orbitDrag.current) {
          orbitDrag.current.active = false;
        }
        orbitDrag.current = null;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hasFrames, frameIndex, frameCount]);

  useEffect(() => {
    if (!autoSpin || !hasFrames) return;
    let raf = 0;
    let last = performance.now();
    const tick = (t: number) => {
      const dt = (t - last) / 1000;
      last = t;
      setYaw01((y) => mod1(y + dt * (1 / 6)));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [autoSpin, hasFrames]);

  const generateReference = async () => {
    if (!refPrompt.trim()) return;
    setGeneratingRef(true);
    try {
      const res = await fetch("/api/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: refPrompt,
          height: 512,
          width: 512,
          steps: 9,
          seed: Math.floor(Math.random() * 1e9),
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const src = data.image_url as string;

      const item: RefItem = {
        id: `gen_${data.job_id ?? crypto.randomUUID()}`,
        name: filenameToName(refPrompt.slice(0, 36) || "Generated"),
        src,
        kind: "generated",
      };

      setCustomRefs((prev) => [item, ...prev]);
      setReference(item.src, item.id);
      setRefPrompt("");
    } catch (e) {
      console.error(e);
    } finally {
      setGeneratingRef(false);
    }
  };

  const runMultiview = async (orbit?: OrbitSpec | null) => {
    if (!objectImage) return;

    setRunning(true);
    resetOutputs();

    try {
      const objBlob = await fetch(objectImage).then((r) => r.blob());

      const refToSend = referenceImage ?? objectImage;
      const refBlob = await fetch(refToSend).then((r) => r.blob());

      const form = new FormData();
      form.append(
        "reference",
        new File([refBlob], "reference.png", { type: "image/png" }),
      );
      form.append(
        "object",
        new File([objBlob], "object.png", { type: "image/png" }),
      );

      form.append("elevation", "10");
      form.append("distance", "2.0");
      form.append("fov", "0.7");
      form.append("steps", "50");
      form.append("max_frames", "21");

      const useOrbit =
        orbit ?? orbitDraft ?? orbitConfirmed ?? orbitFromAngles(0, 0, 0);
      if (useOrbit) {
        form.append("orbit_axis_x", String(useOrbit.axis[0]));
        form.append("orbit_axis_y", String(useOrbit.axis[1]));
        form.append("orbit_axis_z", String(useOrbit.axis[2]));
      }

      const res = await fetch("/api/multiview-transfer", {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error("Multiview generation failed");

      const data = (await res.json()) as MVResponse;

      const nextObjFrames = data.obj_frames ?? [];
      const nextPredFrames = data.pred_frames?.length
        ? data.pred_frames
        : (data.frames ?? []);

      const [readyObjFrames, readyPredFrames] = await Promise.all([
        preloadFrames(nextObjFrames),
        preloadFrames(nextPredFrames),
      ]);

      setObjFrames(readyObjFrames);
      setPredFrames(readyPredFrames);
      setYaw01(0);
    } catch (e) {
      console.error(e);
    } finally {
      setRunning(false);
    }
  };

  const startAxisDrag = (axis: OrbitAxis, clientX: number, clientY: number) => {
    const base = orbitDraft ?? orbitConfirmed ?? orbitFromAngles(0, 0, 0);
    const el = viewportRef.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    orbitDrag.current = {
      active: true,
      axis,
      lastX: clientX,
      lastY: clientY,
      centerX,
      centerY,
      quat: base.quat,
    };

    setActiveOrbitAxis(axis);
    setOrbitDraft(base);

    const onMove = (e: PointerEvent) => {
      const st = orbitDrag.current;
      if (!st?.active) return;

      const dx = e.clientX - st.lastX;
      const dy = e.clientY - st.lastY;

      st.lastX = e.clientX;
      st.lastY = e.clientY;

      const vx = e.clientX - st.centerX;
      const vy = e.clientY - st.centerY;
      const len = Math.hypot(vx, vy) || 1;

      const tangentX = -vy / len;
      const tangentY = vx / len;

      const deltaDeg = (dx * tangentX + dy * tangentY) * 0.85;

      const localAxis: Vec3 =
        st.axis === "x" ? [1, 0, 0] : st.axis === "y" ? [0, 1, 0] : [0, 0, 1];

      const dq = quatFromAxisAngle(localAxis, deltaDeg);

      // post-multiply so rotation stays locked to the gizmo's local axis
      st.quat = quatNormalize(quatMul(st.quat, dq));

      setOrbitDraft(orbitFromQuat(st.quat));
    };

    const onUp = () => {
      const st = orbitDrag.current;
      if (st) {
        setOrbitConfirmed(orbitFromQuat(st.quat));
      }

      setOrbitDraft(null);
      setActiveOrbitAxis(null);
      orbitDrag.current = null;

      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
    window.addEventListener("pointercancel", onUp, { once: true });
  };

  const onNew = () => {
    setObjectImage(null);
    setReferenceImage(null);
    setActiveRefId("none");

    setOrbitDraft(null);
    setOrbitConfirmed(null);
    setActiveOrbitAxis(null);

    resetOutputs();
    setRunning(false);

    if (objInputRef.current) objInputRef.current.value = "";
  };

  const downloadPredictedZip = async () => {
    if (!predFrames.length || savingZip) return;

    setSavingZip(true);
    try {
      const zip = new JSZip();

      await Promise.all(
        predFrames.map(async (url, i) => {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`Failed to fetch frame ${i}: ${url}`);
          const buf = await res.arrayBuffer();
          const name = `pred_${String(i).padStart(3, "0")}.png`;
          zip.file(name, buf);
        }),
      );

      const blob = await zip.generateAsync({ type: "blob" });
      const dlUrl = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = dlUrl;
      a.download = "predicted_frames.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();

      URL.revokeObjectURL(dlUrl);
    } catch (e) {
      console.error(e);
    } finally {
      setSavingZip(false);
    }
  };

  const updateRefScroll01 = () => {
    const el = refStripRef.current;
    if (!el) return;
    const max = Math.max(0, el.scrollWidth - el.clientWidth);
    setRefScroll01(max <= 0 ? 0 : el.scrollLeft / max);
  };

  useEffect(() => {
    updateRefScroll01();
  }, [refs.length]);

  const onRefScrollbarChange = (v01: number) => {
    const el = refStripRef.current;
    if (!el) return;
    const max = Math.max(0, el.scrollWidth - el.clientWidth);
    el.scrollLeft = v01 * max;
    setRefScroll01(v01);
  };

  const refScrollMax = (() => {
    const el = refStripRef.current;
    if (!el) return 0;
    return Math.max(0, el.scrollWidth - el.clientWidth);
  })();

  return (
    <div className="h-screen w-screen bg-[var(--app-bg)] text-[color:var(--app-fg)] flex flex-col overflow-hidden transition-colors duration-200">
      {/* Top bar */}
      <div className="h-14 border-b border-[color:var(--border)] bg-[var(--panel-bg)] backdrop-blur px-4 flex items-center gap-2">
        <div className="text-sm font-semibold tracking-tight">
          Material Transfer Studio
        </div>

        {/* NEW: Top-left actions */}
        <div className="ml-3 flex items-center gap-2">
          <button
            type="button"
            onClick={onNew}
            disabled={running || savingZip}
            className="rounded-full border border-[color:var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[color:var(--muted2)] hover:bg-[var(--surface-hover)] transition disabled:opacity-40 disabled:hover:bg-[var(--surface)] disabled:cursor-not-allowed"
          >
            New
          </button>

          <button
            type="button"
            onClick={downloadPredictedZip}
            disabled={running || savingZip || predFrames.length === 0}
            className="rounded-full border border-[color:var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[color:var(--muted2)] hover:bg-[var(--surface-hover)] transition disabled:opacity-40 disabled:hover:bg-[var(--surface)] disabled:cursor-not-allowed"
            title={
              predFrames.length
                ? "Download predicted frames (.zip)"
                : "No predicted frames yet"
            }
          >
            {savingZip ? "Zipping…" : "Save"}
          </button>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {/* Theme toggle */}
          <button
            type="button"
            onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
            className="rounded-full border border-[color:var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[color:var(--muted2)] hover:bg-[var(--surface-hover)] transition"
            title="Toggle theme"
          >
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </button>

          <button
            type="button"
            onClick={() => setAutoSpin((v) => !v)}
            disabled={!hasFrames}
            className="rounded-full border border-[color:var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[color:var(--muted2)] hover:bg-[var(--surface-hover)] transition disabled:opacity-40 disabled:hover:bg-[var(--surface)] disabled:cursor-not-allowed"
          >
            {autoSpin ? "Stop Spin" : "Auto Spin"}
          </button>

          <button
            type="button"
            onClick={() => setYaw01(0)}
            disabled={!hasFrames}
            className="rounded-full border border-[color:var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[color:var(--muted2)] hover:bg-[var(--surface-hover)] transition disabled:opacity-40 disabled:hover:bg-[var(--surface)] disabled:cursor-not-allowed"
          >
            Reset View
          </button>

          <button
            type="button"
            disabled={!canRun}
            onClick={() => runMultiview(displayOrbit)}
            className="rounded-full bg-[var(--primary-bg)] text-[var(--primary-fg)] px-4 py-2 text-xs font-semibold hover:bg-[var(--primary-bg-hover)] transition disabled:opacity-40 disabled:hover:bg-[var(--primary-bg)] disabled:cursor-not-allowed"
          >
            {running ? "Generating…" : "Generate Multiview"}
          </button>
        </div>
      </div>

      {/* Viewport */}
      <div className="flex-1 relative">
        <div
          ref={viewportRef}
          className="absolute inset-0 bg-[var(--viewport-bg)] overflow-hidden"
          onContextMenu={(e) => e.preventDefault()}
          onPointerDown={(e) => {
            if (e.button === 0 && hasFrames) {
              (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
              beginDrag(e.clientX, e.pointerId);
            }
          }}
          onPointerMove={(e) => {
            moveDrag(e.clientX);
          }}
          onPointerUp={() => {
            endDrag();
          }}
          onPointerCancel={() => {
            endDrag();
          }}
          onWheel={(e) => {
            if (!hasFrames) return;
            e.preventDefault();
            stepFrame(e.deltaY > 0 ? 1 : -1);
          }}
        >
          {/* subtle grid */}
          <div className="pointer-events-none absolute inset-0 opacity-[0.20]">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "linear-gradient(var(--grid-line) 1px, transparent 1px), linear-gradient(90deg, var(--grid-line) 1px, transparent 1px)",
                backgroundSize: "28px 28px",
              }}
            />
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "radial-gradient(circle at center, var(--grid-glow) 0%, var(--grid-falloff) 65%, var(--grid-edge) 100%)",
              }}
            />
          </div>

          {/* Center box + scrub bar */}
          <div className="absolute inset-0 flex items-center justify-center p-4 sm:p-6 md:p-8 min-h-0">
            <div className="flex h-full w-full max-w-[1280px] flex-col items-center justify-center gap-4 min-h-0 min-w-0">
              <button
                type="button"
                onContextMenu={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  pickObject();
                }}
                className={[
                  "relative shrink min-h-0 min-w-0 overflow-hidden rounded-2xl border border-[color:var(--border)] bg-[var(--centerbox-bg)]",
                  "w-full max-w-[1200px] flex-1",
                  objectImage || hasFrames
                    ? "cursor-default"
                    : "cursor-pointer hover:border-[color:var(--border-strong)] transition",
                ].join(" ")}
              >
                <div className="absolute inset-0 flex items-center justify-center p-4 sm:p-5 md:p-6 min-h-0 min-w-0">
                  {currentFrame ? (
                    <img
                      src={currentFrame}
                      alt="frame"
                      className="block max-h-full max-w-full object-contain"
                      draggable={false}
                    />
                  ) : objectImage ? (
                    <img
                      src={objectImage}
                      alt="object"
                      className="block max-h-full max-w-full object-contain"
                      draggable={false}
                    />
                  ) : (
                    <div className="text-[color:var(--muted2)] text-sm">
                      Right-click to upload image
                    </div>
                  )}
                </div>

                <div className="pointer-events-none absolute inset-0 grid place-items-center z-20">
                  {(() => {
                    const boxSize = 320;
                    const axisSize = 268;
                    const orbitRingSize = 188;
                    const gizmoOpacity =
                      isGizmoHovered || activeOrbitAxis !== null ? 1 : 0.28;

                    const { axis: gizmoAxis, deg: gizmoDeg } = quatToAxisAngle(
                      displayOrbit.quat,
                    );
                    const gizmoTransform = `rotate3d(${gizmoAxis[0]}, ${gizmoAxis[1]}, ${gizmoAxis[2]}, ${gizmoDeg}deg)`;

                    const strokeForAxis = (axis: OrbitAxis) => {
                      if (axis === "x") {
                        return activeOrbitAxis === "x"
                          ? "var(--axis-x-active)"
                          : "var(--axis-x)";
                      }
                      if (axis === "y") {
                        return activeOrbitAxis === "y"
                          ? "var(--axis-y-active)"
                          : "var(--axis-y)";
                      }
                      return activeOrbitAxis === "z"
                        ? "var(--axis-z-active)"
                        : "var(--axis-z)";
                    };

                    const widthForAxis = (axis: OrbitAxis) =>
                      activeOrbitAxis === axis ? 5 : 3;

                    return (
                      <div
                        className="relative transition-opacity duration-200 ease-out"
                        style={{
                          width: boxSize,
                          height: boxSize,
                          perspective: "1600px",
                          opacity: gizmoOpacity,
                        }}
                      >
                        <div
                          className="absolute inset-0"
                          style={{
                            transformStyle: "preserve-3d",
                            transform: "rotateX(-24deg) rotateY(30deg)",
                          }}
                        >
                          <div
                            className="absolute inset-0"
                            style={{
                              transformStyle: "preserve-3d",
                              transform: gizmoTransform,
                              willChange: "transform",
                            }}
                          >
                            {/* X axis ring - red */}
                            <div
                              role="button"
                              tabIndex={-1}
                              className="absolute left-1/2 top-1/2 pointer-events-auto"
                              style={{
                                width: axisSize,
                                height: axisSize,
                                transformStyle: "preserve-3d",
                                transform:
                                  "translate3d(-50%, -50%, 0px) rotateY(90deg)",
                                cursor: "grab",
                              }}
                              onPointerEnter={() => setIsGizmoHovered(true)}
                              onPointerLeave={() => setIsGizmoHovered(false)}
                              onPointerDown={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                startAxisDrag("x", e.clientX, e.clientY);
                              }}
                            >
                              <svg
                                width={axisSize}
                                height={axisSize}
                                viewBox={`0 0 ${axisSize} ${axisSize}`}
                                className="overflow-visible"
                              >
                                <circle
                                  cx={axisSize / 2}
                                  cy={axisSize / 2}
                                  r={axisSize / 2 - 6}
                                  fill="none"
                                  stroke={strokeForAxis("x")}
                                  strokeWidth={widthForAxis("x")}
                                  style={{ pointerEvents: "stroke" }}
                                />
                              </svg>
                            </div>

                            {/* Y axis ring - green */}
                            <div
                              role="button"
                              tabIndex={-1}
                              className="absolute left-1/2 top-1/2 pointer-events-auto"
                              style={{
                                width: axisSize,
                                height: axisSize,
                                transformStyle: "preserve-3d",
                                transform:
                                  "translate3d(-50%, -50%, 0px) rotateX(90deg)",
                                cursor: "grab",
                              }}
                              onPointerEnter={() => setIsGizmoHovered(true)}
                              onPointerLeave={() => setIsGizmoHovered(false)}
                              onPointerDown={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                startAxisDrag("y", e.clientX, e.clientY);
                              }}
                            >
                              <svg
                                width={axisSize}
                                height={axisSize}
                                viewBox={`0 0 ${axisSize} ${axisSize}`}
                                className="overflow-visible"
                              >
                                <circle
                                  cx={axisSize / 2}
                                  cy={axisSize / 2}
                                  r={axisSize / 2 - 6}
                                  fill="none"
                                  stroke={strokeForAxis("y")}
                                  strokeWidth={widthForAxis("y")}
                                  style={{ pointerEvents: "stroke" }}
                                />
                              </svg>
                            </div>

                            {/* Z axis ring - blue */}
                            <div
                              role="button"
                              tabIndex={-1}
                              className="absolute left-1/2 top-1/2 pointer-events-auto"
                              style={{
                                width: axisSize,
                                height: axisSize,
                                transformStyle: "preserve-3d",
                                transform: "translate3d(-50%, -50%, 0px)",
                                cursor: "grab",
                              }}
                              onPointerEnter={() => setIsGizmoHovered(true)}
                              onPointerLeave={() => setIsGizmoHovered(false)}
                              onPointerDown={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                startAxisDrag("z", e.clientX, e.clientY);
                              }}
                            >
                              <svg
                                width={axisSize}
                                height={axisSize}
                                viewBox={`0 0 ${axisSize} ${axisSize}`}
                                className="overflow-visible"
                              >
                                <circle
                                  cx={axisSize / 2}
                                  cy={axisSize / 2}
                                  r={axisSize / 2 - 6}
                                  fill="none"
                                  stroke={strokeForAxis("z")}
                                  strokeWidth={widthForAxis("z")}
                                  style={{ pointerEvents: "stroke" }}
                                />
                              </svg>
                            </div>

                            {/* white orbit ring */}
                            <div
                              className="absolute left-1/2 top-1/2 rounded-full"
                              style={{
                                width: orbitRingSize,
                                height: orbitRingSize,
                                border: "2px solid var(--orbit)",
                                transformStyle: "preserve-3d",
                                transform:
                                  "translate3d(-50%, -50%, -4px) rotateX(90deg)",
                                opacity: 0.28,
                              }}
                            />

                            <div
                              className="absolute left-1/2 top-1/2 rounded-full"
                              style={{
                                width: orbitRingSize,
                                height: orbitRingSize,
                                border: "2px solid var(--orbit-strong)",
                                transformStyle: "preserve-3d",
                                transform:
                                  "translate3d(-50%, -50%, 4px) rotateX(90deg)",
                                opacity: 0.96,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              </button>

              {/* frame scrub bar */}
              <div
                className="w-full max-w-[900px] shrink-0"
                onPointerDownCapture={(e) => e.stopPropagation()}
                onPointerMoveCapture={(e) => e.stopPropagation()}
                onPointerUpCapture={(e) => e.stopPropagation()}
                onWheelCapture={(e) => e.stopPropagation()}
              >
                <div className="rounded-full border border-[color:var(--border)] bg-[var(--surface)] backdrop-blur px-3 py-2">
                  <input
                    disabled={!hasFrames}
                    type="range"
                    min={0}
                    max={Math.max(0, frameCount - 1)}
                    step={1}
                    value={frameIndex}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setYaw01(frameCount <= 1 ? 0 : v / (frameCount - 1));
                    }}
                    className="w-full accent-[var(--accent)] disabled:opacity-40"
                  />
                </div>
              </div>
            </div>
          </div>

          {hasFrames && (
            <div className="absolute top-3 left-4 text-[11px] text-[color:var(--muted3)]">
              {frameLabel}
            </div>
          )}

          {running && (
            <div className="absolute inset-0 bg-[var(--overlay-bg)] grid place-items-center">
              <div className="rounded-2xl border border-[color:var(--border)] bg-[var(--toast-bg)] backdrop-blur px-4 py-3 text-sm text-[color:var(--muted2)]">
                Generating…
              </div>
            </div>
          )}
        </div>

        <input
          ref={objInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => onObjectFile(e.target.files?.[0] ?? null)}
        />
      </div>

      {/* Bottom panel */}
      <div className="border-t border-[color:var(--border)] bg-[var(--panel-bg)] backdrop-blur px-4 py-3">
        <div className="mx-auto max-w-[1400px] flex items-center gap-2">
          <input
            value={refPrompt}
            onChange={(e) => setRefPrompt(e.target.value)}
            placeholder="Generate reference…"
            className="flex-1 rounded-xl border border-[color:var(--border)] bg-[var(--input-bg)] px-3 py-2 text-sm text-[color:var(--app-fg)] placeholder:text-[color:var(--placeholder)] outline-none focus:border-[color:var(--border-strong)]"
          />

          <button
            type="button"
            onClick={pickReferenceUpload}
            className="rounded-xl border border-[color:var(--border)] bg-[var(--surface)] px-4 py-2 text-sm text-[color:var(--muted2)] hover:bg-[var(--surface-hover)] transition"
          >
            Upload
          </button>

          <button
            type="button"
            onClick={generateReference}
            disabled={generatingRef || !refPrompt.trim()}
            className="rounded-xl bg-[var(--primary-bg)] text-[var(--primary-fg)] px-4 py-2 text-sm font-semibold hover:bg-[var(--primary-bg-hover)] transition disabled:opacity-40 disabled:hover:bg-[var(--primary-bg)] disabled:cursor-not-allowed"
          >
            {generatingRef ? "Generating…" : "Generate"}
          </button>
        </div>

        <input
          ref={refUploadInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => onReferenceFile(e.target.files?.[0] ?? null)}
        />

        <div className="mx-auto max-w-[1400px] mt-3">
          <div
            ref={refStripRef}
            className="no-scrollbar overflow-x-auto overflow-y-hidden"
            onScroll={updateRefScroll01}
            onWheel={(e) => {
              const el = refStripRef.current;
              if (!el) return;
              const delta =
                Math.abs(e.deltaY) > Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
              el.scrollLeft += delta;
              updateRefScroll01();
              e.preventDefault();
            }}
          >
            <div className="flex gap-3 pb-2 min-w-max">
              <button
                type="button"
                onClick={() => setReference(null, "none")}
                className={[
                  "shrink-0 w-28 rounded-xl border transition overflow-hidden",
                  activeRefId === "none"
                    ? "border-[color:var(--border-strong)] bg-[var(--surface-active)]"
                    : "border-[color:var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-hover)]",
                ].join(" ")}
                title="No reference"
              >
                <div className="aspect-square bg-[var(--thumb-bg)] grid place-items-center text-[11px] text-[color:var(--muted2)]">
                  NONE
                </div>
              </button>

              {refs.map((item) => {
                const active = activeRefId === item.id;
                return (
                  <button
                    type="button"
                    key={item.id}
                    onClick={() => setReference(item.src, item.id)}
                    className={[
                      "shrink-0 w-28 rounded-xl border transition overflow-hidden",
                      active
                        ? "border-[color:var(--border-strong)] bg-[var(--surface-active)]"
                        : "border-[color:var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-hover)]",
                    ].join(" ")}
                    title={item.name}
                  >
                    <div className="aspect-square bg-[var(--thumb-bg)] overflow-hidden">
                      <img
                        src={item.src}
                        alt={item.name}
                        className="w-full h-full object-cover"
                        draggable={false}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {refScrollMax > 0 && (
            <div className="mt-2 rounded-full border border-[color:var(--border)] bg-[var(--surface)] backdrop-blur px-3 py-2">
              <input
                type="range"
                min={0}
                max={1000}
                step={1}
                value={Math.round(refScroll01 * 1000)}
                onChange={(e) =>
                  onRefScrollbarChange(Number(e.target.value) / 1000)
                }
                className="w-full accent-[var(--accent)]"
              />
            </div>
          )}
        </div>
      </div>

      {/* Theme tokens */}
      <style>{`
        :root[data-theme="dark"]{
          color-scheme: dark;
          --app-bg:#0a0b10;
          --app-fg:rgba(255,255,255,1);
          --viewport-bg:#07080d;

          --panel-bg:rgba(255,255,255,0.03);
          --panel-strong:rgba(0,0,0,0.60);

          --surface:rgba(255,255,255,0.05);
          --surface-hover:rgba(255,255,255,0.10);
          --surface-active:rgba(255,255,255,0.10);

          --border:rgba(255,255,255,0.10);
          --border-strong:rgba(255,255,255,0.35);

          --muted:rgba(255,255,255,0.45);
          --muted2:rgba(255,255,255,0.85);
          --muted3:rgba(255,255,255,0.55);
          --placeholder:rgba(255,255,255,0.35);

          --input-bg:rgba(0,0,0,0.35);
          --centerbox-bg:rgba(0,0,0,0.30);
          --thumb-bg:rgba(0,0,0,0.30);

          --overlay-bg:rgba(0,0,0,0.35);
          --toast-bg:rgba(255,255,255,0.10);

          --primary-bg:#ffffff;
          --primary-fg:#000000;
          --primary-bg-hover:rgba(255,255,255,0.90);

          --accent:#ffffff;

          --grid-line:rgba(255,255,255,0.06);
          --grid-glow:rgba(255,255,255,0.10);
          --grid-falloff:rgba(0,0,0,0.90);
          --grid-edge:rgba(0,0,0,1);

          --orbit:rgba(255,255,255,0.70);
          --orbit-strong:rgba(255,255,255,0.80);
          --orbit-dot:rgba(255,255,255,0.80);

          --axis-x:#ff5a5a;
          --axis-y:#48d26f;
          --axis-z:#5aa8ff;

          --axis-x-active:#ff8a8a;
          --axis-y-active:#7ef39d;
          --axis-z-active:#8bc4ff;
        }

        :root[data-theme="light"]{
          color-scheme: light;
          --app-bg:#f6f7fb;
          --app-fg:rgba(17,24,39,1);
          --viewport-bg:#ffffff;

          --panel-bg:rgba(17,24,39,0.04);
          --panel-strong:rgba(255,255,255,0.78);

          --surface:rgba(17,24,39,0.06);
          --surface-hover:rgba(17,24,39,0.10);
          --surface-active:rgba(17,24,39,0.10);

          --border:rgba(17,24,39,0.12);
          --border-strong:rgba(17,24,39,0.35);

          --muted:rgba(17,24,39,0.55);
          --muted2:rgba(17,24,39,0.85);
          --muted3:rgba(17,24,39,0.55);
          --placeholder:rgba(17,24,39,0.40);

          --input-bg:rgba(255,255,255,0.75);
          --centerbox-bg:rgba(255,255,255,0.85);
          --thumb-bg:rgba(255,255,255,0.70);

          --overlay-bg:rgba(255,255,255,0.55);
          --toast-bg:rgba(255,255,255,0.85);

          --primary-bg:rgba(17,24,39,1);
          --primary-fg:#ffffff;
          --primary-bg-hover:rgba(17,24,39,0.90);

          --accent:rgba(17,24,39,1);

          --grid-line:rgba(17,24,39,0.06);
          --grid-glow:rgba(17,24,39,0.08);
          --grid-falloff:rgba(246,247,251,0.75);
          --grid-edge:rgba(246,247,251,1);

          --orbit:rgba(17,24,39,0.55);
          --orbit-strong:rgba(17,24,39,0.70);
          --orbit-dot:rgba(17,24,39,0.70);

          --axis-x:#d63d3d;
          --axis-y:#2fa956;
          --axis-z:#3278d8;

          --axis-x-active:#f06262;
          --axis-y-active:#4dcc73;
          --axis-z-active:#5b98ea;
        }
      `}</style>
    </div>
  );
}

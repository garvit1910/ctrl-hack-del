/**
 * NeuroDetect — Living Neuron 3D Canvas Engine
 *
 * Recursive 3D projected neuron visualization rendered on HTML5 Canvas.
 * Features scroll-to-zoom navigation toward the nucleus with clinical
 * fact reveal upon reaching the center.
 *
 * ─────────────────────────────────────────────────────────────────────
 * 3D Projection & Recursive Branching Logic
 * Inspired by neural web rendering techniques by Sander (sander.land).
 * Original concept: recursive fractal tree with 3D perspective projection.
 * Adapted and rewritten for React/TypeScript with clinical UX integration.
 * ─────────────────────────────────────────────────────────────────────
 *
 * @module components/canvas/HeroNeuron
 */

"use client";

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  project3D,
  rotateX,
  rotateY,
  lerp,
  clamp,
  type Vec3,
} from "@/lib/utils";

/* ─── Clinical Facts ──────────────────────────────────────────────────── */

const CLINICAL_FACTS = [
  {
    title: "Dopamine & Movement",
    body: "Parkinson's disease is caused by the progressive loss of dopaminergic neurons in the substantia nigra, leading to tremor, rigidity, and bradykinesia.",
  },
  {
    title: "Early Detection Matters",
    body: "By the time motor symptoms appear, approximately 60-80% of dopaminergic neurons have already been lost. Early screening tools like spiral analysis can help detect subtle motor changes.",
  },
  {
    title: "The Archimedes Spiral Test",
    body: "Drawing an Archimedes spiral is a validated clinical method. Patients with Parkinson's show increased drawing irregularity, slower speeds, and higher radial deviation from the ideal spiral.",
  },
  {
    title: "Neural Communication",
    body: "Each neuron connects to up to 10,000 others through synapses. In Parkinson's, the loss of these connections in the basal ganglia disrupts the motor circuit.",
  },
  {
    title: "Micrographia",
    body: "Progressive reduction in handwriting size (micrographia) is one of the earliest motor signs of Parkinson's disease, often appearing years before diagnosis.",
  },
];

/* ─── Branch Node Type ─────────────────────────────────────────────────── */

interface BranchNode {
  start: Vec3;
  end: Vec3;
  depth: number;
  children: BranchNode[];
  pulsePhase: number;
}

/* ─── Neuron Tree Generator ────────────────────────────────────────────── */

function buildNeuronTree(
  origin: Vec3,
  dir: Vec3,
  length: number,
  depth: number,
  maxDepth: number
): BranchNode {
  const end: Vec3 = {
    x: origin.x + dir.x * length,
    y: origin.y + dir.y * length,
    z: origin.z + dir.z * length,
  };

  const children: BranchNode[] = [];

  if (depth < maxDepth) {
    const numBranches = depth < 2 ? 3 : 2;
    for (let i = 0; i < numBranches; i++) {
      const spread = 0.45 + depth * 0.05;
      const offset = spread * (i - (numBranches - 1) / 2);
      const jitter = (Math.random() - 0.5) * 0.25;

      let newDir = rotateY(dir, offset + jitter);
      newDir = rotateX(newDir, offset * 0.6 + jitter * 0.5);

      const mag = Math.sqrt(
        newDir.x ** 2 + newDir.y ** 2 + newDir.z ** 2
      );
      const normalized: Vec3 = {
        x: newDir.x / mag,
        y: newDir.y / mag,
        z: newDir.z / mag,
      };

      children.push(
        buildNeuronTree(
          end,
          normalized,
          length * 0.7,
          depth + 1,
          maxDepth
        )
      );
    }
  }

  return {
    start: origin,
    end,
    depth,
    children,
    pulsePhase: Math.random() * Math.PI * 2,
  };
}

/* ─── Synapse Particle ─────────────────────────────────────────────────── */

interface Particle {
  branch: BranchNode;
  t: number;
  speed: number;
  size: number;
}

/* ─── Main Component ──────────────────────────────────────────────────── */

export default function HeroNeuron() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  const [zoom, setZoom] = useState(0);
  const [showDialog, setShowDialog] = useState(false);
  const [factIndex, setFactIndex] = useState(0);
  const [hasReachedNucleus, setHasReachedNucleus] = useState(false);

  // Build neuron tree once using multiple root directions
  const trees = useMemo(() => {
    const directions: Vec3[] = [
      { x: 0, y: -1, z: 0 },
      { x: 0.7, y: -0.7, z: 0 },
      { x: -0.7, y: -0.7, z: 0 },
      { x: 0, y: -0.7, z: 0.7 },
      { x: 0, y: -0.7, z: -0.7 },
      { x: 0, y: 1, z: 0 },
      { x: 0.5, y: 0.85, z: 0 },
      { x: -0.5, y: 0.85, z: 0 },
    ];

    return directions.map((d) =>
      buildNeuronTree({ x: 0, y: 0, z: 0 }, d, 120, 0, 7)
    );
  }, []);

  // Generate particles along branches
  const particles = useMemo(() => {
    const parts: Particle[] = [];
    function collect(node: BranchNode) {
      if (Math.random() < 0.3) {
        parts.push({
          branch: node,
          t: Math.random(),
          speed: 0.002 + Math.random() * 0.004,
          size: 1 + Math.random() * 2,
        });
      }
      node.children.forEach(collect);
    }
    trees.forEach(collect);
    return parts;
  }, [trees]);

  /* ─── Scroll Zoom Handler ──────────────────────────────────── */

  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      // Faster scroll sensitivity for a responsive zoom feel
      const delta = e.deltaY * -0.003;
      setZoom((prev) => {
        const next = clamp(prev + delta, 0, 1);

        // Trigger dialog at nucleus
        if (next >= 0.95 && !hasReachedNucleus) {
          setHasReachedNucleus(true);
          setFactIndex(Math.floor(Math.random() * CLINICAL_FACTS.length));
          setShowDialog(true);
        }

        return next;
      });
    },
    [hasReachedNucleus]
  );

  /* ─── Canvas Setup & Animation Loop ────────────────────────── */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener("wheel", handleWheel, { passive: false });

    const ctx = canvas.getContext("2d")!;
    let width = 0;
    let height = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      width = canvas!.clientWidth;
      height = canvas!.clientHeight;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    resize();
    window.addEventListener("resize", resize);

    /* ─── Render Branch Recursively ────────────────────────── */

    function renderBranch(
      node: BranchNode,
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      globalAlpha: number,
      cameraZ: number
    ) {
      // Rotate points
      let s = rotateY(node.start, rotAngleY);
      s = rotateX(s, rotAngleX);
      let e = rotateY(node.end, rotAngleY);
      e = rotateX(e, rotAngleX);

      // Move scene toward camera (fly into the neuron)
      s = { ...s, z: s.z + cameraZ };
      e = { ...e, z: e.z + cameraZ };

      const p1 = project3D(s, focalLength, cx, cy);
      const p2 = project3D(e, focalLength, cx, cy);

      // Pulse brightness
      const pulse =
        0.6 + 0.4 * Math.sin(time * 2 + node.pulsePhase);

      // Depth-based color: cyan core → blue tips
      const depthRatio = node.depth / 7;
      const r = Math.round(lerp(0, 80, depthRatio));
      const g = Math.round(lerp(220, 140, depthRatio));
      const b = Math.round(lerp(255, 255, depthRatio));

      const lineWidth = Math.max(0.5, (1 - depthRatio) * 3 * p1.scale);
      const alpha = clamp(
        globalAlpha * pulse * (1 - depthRatio * 0.4) * p1.scale,
        0,
        1
      );

      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.stroke();

      // Render children
      node.children.forEach((child) =>
        renderBranch(
          child,
          time,
          focalLength,
          cx,
          cy,
          rotAngleY,
          rotAngleX,
          globalAlpha,
          cameraZ
        )
      );
    }

    /* ─── Render Nucleus ───────────────────────────────────── */

    function renderNucleus(
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      zoomLevel: number,
      cameraZ: number
    ) {
      let nucleusPos: Vec3 = { x: 0, y: 0, z: 0 };
      nucleusPos = rotateY(nucleusPos, rotAngleY);
      nucleusPos = rotateX(nucleusPos, rotAngleX);
      nucleusPos = { ...nucleusPos, z: nucleusPos.z + cameraZ };

      const projected = project3D(nucleusPos, focalLength, cx, cy);
      const baseRadius = 12 + zoomLevel * 30;
      const pulse = 1 + 0.15 * Math.sin(time * 3);
      const radius = baseRadius * projected.scale * pulse;

      // Outer glow
      const glowGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        radius * 0.3,
        projected.x,
        projected.y,
        radius * 2.5
      );
      glowGrad.addColorStop(0, `rgba(0, 220, 255, ${0.3 * pulse})`);
      glowGrad.addColorStop(0.5, `rgba(0, 180, 255, ${0.1 * pulse})`);
      glowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");

      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = glowGrad;
      ctx.fill();

      // Core
      const coreGrad = ctx.createRadialGradient(
        projected.x,
        projected.y,
        0,
        projected.x,
        projected.y,
        radius
      );
      coreGrad.addColorStop(0, "rgba(200, 255, 255, 0.95)");
      coreGrad.addColorStop(0.4, "rgba(0, 220, 255, 0.8)");
      coreGrad.addColorStop(1, "rgba(0, 100, 200, 0.2)");

      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = coreGrad;
      ctx.fill();
    }

    /* ─── Render Particles ─────────────────────────────────── */

    function renderParticles(
      time: number,
      focalLength: number,
      cx: number,
      cy: number,
      rotAngleY: number,
      rotAngleX: number,
      cameraZ: number
    ) {
      particles.forEach((p) => {
        p.t = (p.t + p.speed) % 1;

        const pos: Vec3 = {
          x: lerp(p.branch.start.x, p.branch.end.x, p.t),
          y: lerp(p.branch.start.y, p.branch.end.y, p.t),
          z: lerp(p.branch.start.z, p.branch.end.z, p.t),
        };

        let rPos = rotateY(pos, rotAngleY);
        rPos = rotateX(rPos, rotAngleX);
        rPos = { ...rPos, z: rPos.z + cameraZ };

        const proj = project3D(rPos, focalLength, cx, cy);
        const alpha = 0.4 + 0.6 * Math.sin(time * 5 + p.branch.pulsePhase);

        ctx.beginPath();
        ctx.arc(proj.x, proj.y, p.size * proj.scale, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 255, 220, ${clamp(alpha * proj.scale, 0, 1)})`;
        ctx.fill();
      });
    }

    /* ─── Animation Frame ──────────────────────────────────── */

    let currentZoom = 0;

    function animate() {
      const time = (timeRef.current += 0.016);

      // Smooth zoom interpolation
      currentZoom = lerp(currentZoom, zoom, 0.08);

      ctx.clearRect(0, 0, width, height);

      // Background radial gradient
      const bgGrad = ctx.createRadialGradient(
        width / 2,
        height / 2,
        0,
        width / 2,
        height / 2,
        width * 0.7
      );
      bgGrad.addColorStop(0, "rgba(5, 15, 25, 1)");
      bgGrad.addColorStop(1, "rgba(0, 0, 0, 1)");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const baseFocal = 400;
      const focalLength = baseFocal;

      // Camera flies forward into the neuron as zoom increases
      // Negative Z = camera moving toward origin (the nucleus)
      const cameraZ = currentZoom * 350;

      // Slow rotation as user zooms in (feels more controlled)
      const rotSpeed = lerp(0.15, 0.03, currentZoom);
      const rotY = time * rotSpeed;
      const rotXAngle = Math.sin(time * 0.1) * lerp(0.15, 0.03, currentZoom);

      // Render neuron branches
      trees.forEach((tree) => {
        renderBranch(tree, time, focalLength, cx, cy, rotY, rotXAngle, 0.9, cameraZ);
      });

      // Render nucleus
      renderNucleus(time, focalLength, cx, cy, rotY, rotXAngle, currentZoom, cameraZ);

      // Render synapse particles
      renderParticles(time, focalLength, cx, cy, rotY, rotXAngle, cameraZ);

      // Zoom indicator
      if (currentZoom > 0.01) {
        const indicatorText = `ZOOM: ${Math.round(currentZoom * 100)}%`;
        ctx.font = "12px monospace";
        ctx.fillStyle = `rgba(0, 220, 255, ${0.4 + currentZoom * 0.4})`;
        ctx.textAlign = "right";
        ctx.fillText(indicatorText, width - 20, height - 20);

        if (currentZoom < 0.9) {
          ctx.font = "11px monospace";
          ctx.fillStyle = "rgba(0, 220, 255, 0.3)";
          ctx.textAlign = "center";
          ctx.fillText(
            "Scroll to dive into the nucleus...",
            cx,
            height - 20
          );
        }
      }

      animFrameRef.current = requestAnimationFrame(animate);
    }

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("wheel", handleWheel);
    };
  }, [trees, particles, zoom, handleWheel]);

  /* ─── Reset on dialog close ──────────────────────────────── */

  const handleDialogClose = useCallback((open: boolean) => {
    setShowDialog(open);
    if (!open) {
      setTimeout(() => setHasReachedNucleus(false), 1000);
    }
  }, []);

  const fact = CLINICAL_FACTS[factIndex];

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        style={{ touchAction: "none" }}
      />

      {/* Scroll hint overlay */}
      {zoom < 0.05 && (
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-pulse">
          <span className="text-xs text-cyan-400/60 font-mono tracking-widest uppercase">
            Scroll to explore
          </span>
          <div className="w-5 h-8 border-2 border-cyan-500/30 rounded-full flex justify-center pt-1">
            <div className="w-1 h-2 bg-cyan-400/60 rounded-full animate-bounce" />
          </div>
        </div>
      )}

      {/* Clinical fact dialog */}
      <Dialog open={showDialog} onOpenChange={handleDialogClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              {fact.title}
            </DialogTitle>
            <DialogDescription className="pt-2 text-zinc-300 leading-relaxed">
              {fact.body}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end pt-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setFactIndex((prev) => (prev + 1) % CLINICAL_FACTS.length);
              }}
            >
              Next Fact
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

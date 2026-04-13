import React, { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Line, Text } from '@react-three/drei';
import * as THREE from 'three';
import { API_BASE_URL } from '../config';

// ── Constants ────────────────────────────────────────────────────────────────
const BODY_COLORS = {
  star:       '#FFD166',
  black_hole: '#EE6055',
  planet:     '#06D6A0',
  moon:       '#8BE8FD',
  asteroid:   '#ADB5BD',
  spacecraft: '#A78BFA',
  unknown:    '#94A3B8',
};

const SPEED_OPTIONS = [0.5, 1, 2, 5, 10];

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Normalize all snapshot positions into a [-viewRadius, viewRadius] cube. */
function normalizeTrajectory(snapshots, meta, viewRadius = 40) {
  if (!snapshots || snapshots.length === 0) return { snapshots: [], scale: 1 };

  let maxDist = 0;
  for (const snap of snapshots) {
    for (const b of snap.bodies) {
      const d = Math.sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
      if (d > maxDist) maxDist = d;
    }
  }

  const scale = maxDist > 0 ? viewRadius / maxDist : 1;

  const normalized = snapshots.map(snap => ({
    time: snap.time,
    bodies: snap.bodies.map(b => ({
      name: b.name,
      x: b.x * scale,
      y: b.y * scale,
      z: b.z * scale,
      speed: b.speed,
    })),
  }));

  return { snapshots: normalized, scale };
}

/** Return a log-scaled sphere radius for display. */
function bodyDisplayRadius(mass, type) {
  if (!mass || mass <= 0) return 0.3;
  const logMass = Math.log10(mass);
  const minLog = 20;
  const maxLog = 32;
  const clamped = Math.max(minLog, Math.min(maxLog, logMass));
  const t = (clamped - minLog) / (maxLog - minLog);
  if (type === 'star') return 0.8 + t * 2.0;
  if (type === 'black_hole') return 0.6 + t * 1.8;
  return 0.2 + t * 1.2;
}

function getColor(meta, name) {
  if (meta[name]?.color) return meta[name].color;
  const type = meta[name]?.type || 'unknown';
  return BODY_COLORS[type] || BODY_COLORS.unknown;
}

function formatTime(s) {
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${(s / 60).toFixed(1)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} hr`;
  if (s < 31536000) return `${(s / 86400).toFixed(1)} day`;
  return `${(s / 31536000).toFixed(2)} yr`;
}

// ── 3D Body Sphere ───────────────────────────────────────────────────────────
function BodySphere({ position, color, radius, name, showLabel, emissive }) {
  const meshRef = useRef();
  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive ? color : '#000000'}
          emissiveIntensity={emissive ? 0.6 : 0}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>
      {/* Glow ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 1.2, radius * 1.5, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>
      {showLabel && (
        <Text
          position={[0, radius + 0.8, 0]}
          fontSize={0.5}
          color="#E2E8F0"
          anchorX="center"
          anchorY="bottom"
          font="https://fonts.gstatic.com/s/inter/v18/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfAZ9hiA.woff2"
        >
          {name}
        </Text>
      )}
    </group>
  );
}

// ── Orbit Trail ──────────────────────────────────────────────────────────────
function OrbitTrail({ points, color }) {
  if (!points || points.length < 2) return null;
  return (
    <Line
      points={points}
      color={color}
      lineWidth={1.2}
      transparent
      opacity={0.5}
    />
  );
}

// ── Auto-rotate camera for mini mode ─────────────────────────────────────────
function AutoRotate({ enabled, speed = 0.3 }) {
  const { camera } = useThree();
  useFrame((_, delta) => {
    if (!enabled) return;
    const angle = speed * delta;
    const x = camera.position.x;
    const z = camera.position.z;
    camera.position.x = x * Math.cos(angle) - z * Math.sin(angle);
    camera.position.z = x * Math.sin(angle) + z * Math.cos(angle);
    camera.lookAt(0, 0, 0);
  });
  return null;
}

// ── Animated Scene ───────────────────────────────────────────────────────────
function AnimatedScene({ snapshots, meta, isFullMode, playbackRef }) {
  const bodyPositions = useRef({});
  const trailPaths = useRef({});

  const fullTrails = useMemo(() => {
    const trails = {};
    for (const name of Object.keys(meta)) trails[name] = [];
    for (const snap of snapshots) {
      for (const b of snap.bodies) {
        if (!trails[b.name]) trails[b.name] = [];
        trails[b.name].push([b.x, b.y, b.z]);
      }
    }
    return trails;
  }, [snapshots, meta]);

  useFrame(() => {
    if (!snapshots.length) return;
    const pb = playbackRef.current;
    if (!pb) return;
    const idx = Math.min(Math.floor(pb.frameIndex), snapshots.length - 1);
    const snap = snapshots[idx];
    if (!snap) return;

    const positions = {};
    for (const b of snap.bodies) positions[b.name] = [b.x, b.y, b.z];
    bodyPositions.current = positions;

    const partials = {};
    for (const name of Object.keys(meta)) {
      const full = fullTrails[name];
      if (full && full.length > 0) {
        partials[name] = full.slice(0, Math.min(idx + 1, full.length));
      }
    }
    trailPaths.current = partials;
  });

  return (
    <AnimatedBodies
      meta={meta}
      bodyPositions={bodyPositions}
      trailPaths={trailPaths}
      isFullMode={isFullMode}
    />
  );
}

function AnimatedBodies({ meta, bodyPositions, trailPaths, isFullMode }) {
  const [, forceUpdate] = useState(0);
  useFrame(() => forceUpdate(c => c + 1));

  const names = Object.keys(meta);
  const positions = bodyPositions.current;
  const trails = trailPaths.current;

  return (
    <>
      {names.map(name => {
        const pos = positions[name];
        if (!pos) return null;
        const color = getColor(meta, name);
        const type = meta[name]?.type || 'unknown';
        const mass = meta[name]?.mass || 1e24;
        const radius = bodyDisplayRadius(mass, type);
        const isEmissive = type === 'star' || type === 'black_hole';

        return (
          <React.Fragment key={name}>
            <BodySphere
              position={pos}
              color={color}
              radius={radius}
              name={name}
              showLabel={isFullMode}
              emissive={isEmissive}
            />
            {trails[name] && trails[name].length > 1 && (
              <OrbitTrail points={trails[name]} color={color} />
            )}
          </React.Fragment>
        );
      })}
    </>
  );
}

// ── Playback Controls ────────────────────────────────────────────────────────
function PlaybackControls({
  currentFrame, totalFrames, isPlaying, speed,
  onPlayPause, onScrub, onSpeedChange, currentTime, totalTime,
}) {
  return (
    <div className="viz-controls">
      <button className="viz-play-btn" onClick={onPlayPause} title={isPlaying ? 'Pause' : 'Play'}>
        {isPlaying ? '\u23F8' : '\u25B6'}
      </button>
      <div className="viz-time-label">{formatTime(currentTime)}</div>
      <input
        className="viz-scrubber"
        type="range"
        min={0}
        max={totalFrames - 1}
        value={currentFrame}
        onChange={(e) => onScrub(Number(e.target.value))}
      />
      <div className="viz-time-label">{formatTime(totalTime)}</div>
      <div className="viz-speed-group">
        {SPEED_OPTIONS.map(s => (
          <button
            key={s}
            className={`viz-speed-btn ${speed === s ? 'active' : ''}`}
            onClick={() => onSpeedChange(s)}
          >
            {s}\u00D7
          </button>
        ))}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// Main OrbitVisualizer Component
//
// KEY FIX: Uses a SINGLE <Canvas> that is ALWAYS mounted. Mini vs Full mode
//          is toggled via CSS class on the wrapper div — the canvas DOM node
//          is never destroyed, so the WebGL context is never lost.
// ══════════════════════════════════════════════════════════════════════════════
export default function OrbitVisualizer({ visible, onClose }) {
  const [isFullMode, setIsFullMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);
  const playbackRef = useRef({ frameIndex: 0, accumulator: 0 });
  const animFrameRef = useRef(null);

  // Fetch trajectory data
  const fetchTrajectory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/trajectory`, { credentials: 'include' });
      if (!res.ok) throw new Error('Failed to fetch');
      const data = await res.json();
      if (data.available && data.snapshots.length > 0) {
        const { snapshots, scale } = normalizeTrajectory(data.snapshots, data.meta);
        setTrajectoryData({ snapshots, meta: data.meta, scale });
        playbackRef.current = { frameIndex: 0, accumulator: 0 };
        setCurrentFrame(0);
        setIsPlaying(true);
      } else {
        setTrajectoryData(null);
      }
    } catch (err) {
      console.error('Trajectory fetch error:', err);
      setTrajectoryData(null);
    }
    setLoading(false);
  }, []);

  // Fetch on show
  useEffect(() => {
    if (visible) fetchTrajectory();
  }, [visible, fetchTrajectory]);

  // Playback animation loop
  useEffect(() => {
    if (!visible || !trajectoryData || !isPlaying) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const totalFrames = trajectoryData.snapshots.length;
    let lastTime = performance.now();

    const tick = (now) => {
      const delta = (now - lastTime) / 1000;
      lastTime = now;
      const framesPerSec = 30 * speed;
      playbackRef.current.frameIndex += framesPerSec * delta;
      if (playbackRef.current.frameIndex >= totalFrames) {
        playbackRef.current.frameIndex = 0;
      }
      setCurrentFrame(Math.floor(playbackRef.current.frameIndex));
      animFrameRef.current = requestAnimationFrame(tick);
    };

    animFrameRef.current = requestAnimationFrame(tick);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [visible, trajectoryData, isPlaying, speed]);

  // Keyboard: Escape to close/minimize
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        if (isFullMode) setIsFullMode(false);
        else if (onClose) onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isFullMode, onClose]);

  // Collapse when parent hides
  useEffect(() => {
    if (!visible) setIsFullMode(false);
  }, [visible]);

  if (!visible) return null;

  const totalFrames = trajectoryData?.snapshots?.length || 0;
  const currentTime = trajectoryData?.snapshots?.[currentFrame]?.time || 0;
  const totalTime   = trajectoryData?.snapshots?.[totalFrames - 1]?.time || 0;

  const handleScrub = (frame) => {
    playbackRef.current.frameIndex = frame;
    setCurrentFrame(frame);
  };

  // ────────────────────────────────────────────────────────────────────────
  // Render: One wrapper, one Canvas. The wrapper class toggles mini ↔ full.
  // ────────────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Dark backdrop — only visible in full mode */}
      {isFullMode && (
        <div
          className="viz-backdrop"
          onClick={() => setIsFullMode(false)}
        />
      )}

      {/* Single persistent panel */}
      <div className={isFullMode ? 'viz-panel viz-panel--full' : 'viz-panel viz-panel--mini'}>
        {/* Header (full mode only) */}
        {isFullMode && (
          <div className="viz-full-header">
            <span className="viz-full-title">{'\u269B'} ORBIT VISUALIZER</span>
            <div className="viz-header-actions">
              <button className="viz-minimize-btn" onClick={() => setIsFullMode(false)} title="Minimize">
                {'\u25BC'}
              </button>
              <button className="viz-close-btn" onClick={onClose} title="Close">
                {'\u2715'}
              </button>
            </div>
          </div>
        )}

        {/* Canvas area — ALWAYS renders the same Canvas */}
        <div className="viz-canvas-wrapper">
          {loading ? (
            <div className="viz-loading-full">
              <div className="inspector-spinner" />
              <span>Loading trajectory data...</span>
            </div>
          ) : !trajectoryData ? (
            <div className="viz-loading-full">
              <span>
                {isFullMode
                  ? 'No trajectory available. Run a simulation with visualize=on first.'
                  : 'No trajectory data'}
              </span>
            </div>
          ) : (
            <Canvas
              camera={{ position: [30, 20, 30], fov: 50 }}
              gl={{
                antialias: true,
                alpha: true,
                powerPreference: 'high-performance',
                preserveDrawingBuffer: true,
              }}
              resize={{ debounce: 0 }}
              onCreated={({ gl }) => {
                gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
              }}
            >
              <color attach="background" args={['#060910']} />
              <ambientLight intensity={0.4} />
              <pointLight position={[0, 0, 0]} intensity={1.8} color="#FFD166" />
              {isFullMode && <directionalLight position={[20, 30, 10]} intensity={0.3} />}
              <Stars
                radius={isFullMode ? 100 : 80}
                depth={isFullMode ? 50 : 40}
                count={isFullMode ? 2000 : 800}
                factor={isFullMode ? 4 : 3}
                saturation={0.3}
                fade
                speed={0.4}
              />
              <AutoRotate enabled={!isFullMode} speed={0.4} />
              {isFullMode && (
                <OrbitControls
                  enablePan
                  enableZoom
                  enableRotate
                  dampingFactor={0.1}
                  enableDamping
                  minDistance={5}
                  maxDistance={200}
                />
              )}
              {isFullMode && (
                <gridHelper args={[100, 20, '#1a2030', '#101620']} position={[0, -15, 0]} />
              )}
              <AnimatedScene
                snapshots={trajectoryData.snapshots}
                meta={trajectoryData.meta}
                isFullMode={isFullMode}
                playbackRef={playbackRef}
              />
            </Canvas>
          )}
        </div>

        {/* Mini overlay (click to expand) */}
        {!isFullMode && trajectoryData && (
          <div className="viz-mini-overlay" onClick={() => setIsFullMode(true)}>
            <span className="viz-mini-label">{'\u25B6'} ORBIT REPLAY</span>
            <span className="viz-mini-hint">Click to expand</span>
          </div>
        )}

        {/* Full-mode playback controls */}
        {isFullMode && trajectoryData && totalFrames > 0 && (
          <PlaybackControls
            currentFrame={currentFrame}
            totalFrames={totalFrames}
            isPlaying={isPlaying}
            speed={speed}
            onPlayPause={() => setIsPlaying(p => !p)}
            onScrub={handleScrub}
            onSpeedChange={setSpeed}
            currentTime={currentTime}
            totalTime={totalTime}
          />
        )}
      </div>
    </>
  );
}

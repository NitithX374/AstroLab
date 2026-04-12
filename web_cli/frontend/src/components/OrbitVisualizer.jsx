import React, { useRef, useState, useMemo, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Line, Text, Html } from '@react-three/drei';
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
  // log10(mass) ranges from ~22 (moon) to ~30 (star)
  const logMass = Math.log10(mass);
  const minLog = 20;
  const maxLog = 32;
  const clamped = Math.max(minLog, Math.min(maxLog, logMass));
  const t = (clamped - minLog) / (maxLog - minLog);
  // Stars are bigger, planets medium, moons small
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
function AnimatedScene({
  snapshots, meta, isFullMode, playbackRef,
}) {
  const bodyPositions = useRef({});
  const trailPaths = useRef({});

  // Build full trail paths from all snapshots per body
  const fullTrails = useMemo(() => {
    const trails = {};
    const names = Object.keys(meta);
    for (const name of names) {
      trails[name] = [];
    }
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

    // Update positions
    const positions = {};
    for (const b of snap.bodies) {
      positions[b.name] = [b.x, b.y, b.z];
    }
    bodyPositions.current = positions;

    // Build partial trails up to current frame
    const partials = {};
    const names = Object.keys(meta);
    for (const name of names) {
      const full = fullTrails[name];
      if (full && full.length > 0) {
        partials[name] = full.slice(0, Math.min(idx + 1, full.length));
      }
    }
    trailPaths.current = partials;
  });

  // We need to render bodies — use a component that re-renders via useFrame
  return <AnimatedBodies
    meta={meta}
    bodyPositions={bodyPositions}
    trailPaths={trailPaths}
    isFullMode={isFullMode}
  />;
}

function AnimatedBodies({ meta, bodyPositions, trailPaths, isFullMode }) {
  const [, forceUpdate] = useState(0);

  useFrame(() => {
    forceUpdate(c => c + 1);
  });

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
        {isPlaying ? '⏸' : '▶'}
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
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Main OrbitVisualizer Component ───────────────────────────────────────────
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

  // Playback animation loop (runs outside React render cycle for performance)
  useEffect(() => {
    if (!visible || !trajectoryData || !isPlaying) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const totalFrames = trajectoryData.snapshots.length;
    let lastTime = performance.now();

    const tick = (now) => {
      const delta = (now - lastTime) / 1000; // seconds
      lastTime = now;

      // advance ~30 frames per second at 1× speed
      const framesPerSec = 30 * speed;
      playbackRef.current.frameIndex += framesPerSec * delta;

      if (playbackRef.current.frameIndex >= totalFrames) {
        playbackRef.current.frameIndex = 0; // loop
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

  if (!visible) return null;

  const totalFrames = trajectoryData?.snapshots?.length || 0;
  const currentTime = trajectoryData?.snapshots?.[currentFrame]?.time || 0;
  const totalTime = trajectoryData?.snapshots?.[totalFrames - 1]?.time || 0;

  const handleScrub = (frame) => {
    playbackRef.current.frameIndex = frame;
    setCurrentFrame(frame);
  };

  // ── Mini mode ──────────────────────────────────────────────────────────
  if (!isFullMode) {
    return (
      <div className="viz-mini" onClick={() => setIsFullMode(true)}>
        {loading ? (
          <div className="viz-mini-loading">
            <div className="inspector-spinner" />
            <span>Loading trajectory…</span>
          </div>
        ) : !trajectoryData ? (
          <div className="viz-mini-empty">
            <span>No trajectory data</span>
          </div>
        ) : (
          <>
            <Canvas
              camera={{ position: [30, 20, 30], fov: 50 }}
              style={{ borderRadius: '10px' }}
              gl={{ antialias: true, alpha: true }}
            >
              <color attach="background" args={['#080b12']} />
              <ambientLight intensity={0.4} />
              <pointLight position={[0, 0, 0]} intensity={1.5} color="#FFD166" />
              <Stars radius={80} depth={40} count={800} factor={3} saturation={0.2} fade speed={0.5} />
              <AutoRotate enabled={true} speed={0.4} />
              <AnimatedScene
                snapshots={trajectoryData.snapshots}
                meta={trajectoryData.meta}
                isFullMode={false}
                playbackRef={playbackRef}
              />
            </Canvas>
            <div className="viz-mini-overlay">
              <span className="viz-mini-label">▶ ORBIT REPLAY</span>
              <span className="viz-mini-hint">Click to expand</span>
            </div>
          </>
        )}
      </div>
    );
  }

  // ── Full mode ──────────────────────────────────────────────────────────
  return (
    <div className="viz-full-overlay" onClick={(e) => {
      if (e.target === e.currentTarget) setIsFullMode(false);
    }}>
      <div className="viz-full-container">
        {/* Header */}
        <div className="viz-full-header">
          <span className="viz-full-title">⚛ ORBIT VISUALIZER</span>
          <div className="viz-header-actions">
            <button className="viz-minimize-btn" onClick={() => setIsFullMode(false)} title="Minimize">
              ▼
            </button>
            <button className="viz-close-btn" onClick={onClose} title="Close">
              ✕
            </button>
          </div>
        </div>

        {/* 3D Canvas */}
        <div className="viz-canvas-wrapper">
          {loading ? (
            <div className="viz-loading-full">
              <div className="inspector-spinner" />
              <span>Loading trajectory data…</span>
            </div>
          ) : !trajectoryData ? (
            <div className="viz-loading-full">
              <span>No trajectory available. Run a simulation with <code>visualize=on</code> first.</span>
            </div>
          ) : (
            <Canvas
              camera={{ position: [45, 30, 45], fov: 50 }}
              gl={{ antialias: true, alpha: true }}
            >
              <color attach="background" args={['#060910']} />
              <ambientLight intensity={0.35} />
              <pointLight position={[0, 0, 0]} intensity={2} color="#FFD166" />
              <directionalLight position={[20, 30, 10]} intensity={0.3} />
              <Stars radius={100} depth={50} count={2000} factor={4} saturation={0.3} fade speed={0.3} />
              <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                dampingFactor={0.1}
                enableDamping
                minDistance={5}
                maxDistance={200}
              />
              <gridHelper args={[100, 20, '#1a2030', '#101620']} position={[0, -15, 0]} />
              <AnimatedScene
                snapshots={trajectoryData.snapshots}
                meta={trajectoryData.meta}
                isFullMode={true}
                playbackRef={playbackRef}
              />
            </Canvas>
          )}
        </div>

        {/* Playback Controls */}
        {trajectoryData && totalFrames > 0 && (
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
    </div>
  );
}

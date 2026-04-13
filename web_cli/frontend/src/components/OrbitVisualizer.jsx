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

// ══════════════════════════════════════════════════════════════════════════════
// SceneContent — renders bodies + trails using IMPERATIVE ref updates.
//
// IMPORTANT: NO useState/forceUpdate inside useFrame.
// Body positions are updated by directly mutating group.position via refs.
// Trails are updated only when `currentFrame` prop changes (React re-render
// driven by parent, NOT by useFrame).
// ══════════════════════════════════════════════════════════════════════════════
function SceneContent({ snapshots, meta, isFullMode, playbackRef, currentFrame }) {
  const bodyRefs = useRef({});

  // Pre-compute all trail paths once
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

  // Compute visible trails based on currentFrame (driven by parent state)
  const visibleTrails = useMemo(() => {
    const result = {};
    const idx = Math.min(currentFrame, snapshots.length - 1);
    for (const name of Object.keys(meta)) {
      const full = fullTrails[name];
      if (full && full.length > 1) {
        result[name] = full.slice(0, Math.min(idx + 1, full.length));
      }
    }
    return result;
  }, [currentFrame, fullTrails, meta, snapshots.length]);

  // Imperative position updates — NO React re-renders
  useFrame(() => {
    if (!snapshots.length) return;
    const pb = playbackRef.current;
    if (!pb) return;
    const idx = Math.min(Math.floor(pb.frameIndex), snapshots.length - 1);
    const snap = snapshots[idx];
    if (!snap) return;

    for (const b of snap.bodies) {
      const group = bodyRefs.current[b.name];
      if (group) {
        group.position.set(b.x, b.y, b.z);
      }
    }
  });

  const names = Object.keys(meta);

  return (
    <group>
      {names.map(name => {
        const color = getColor(meta, name);
        const type = meta[name]?.type || 'unknown';
        const mass = meta[name]?.mass || 1e24;
        const radius = bodyDisplayRadius(mass, type);
        const isEmissive = type === 'star' || type === 'black_hole';

        // Initial position from first snapshot
        const firstSnap = snapshots[0];
        const firstBody = firstSnap?.bodies?.find(b => b.name === name);
        const initPos = firstBody ? [firstBody.x, firstBody.y, firstBody.z] : [0, 0, 0];

        return (
          <React.Fragment key={name}>
            {/* Body group — position is mutated by useFrame via ref */}
            <group
              ref={el => { if (el) bodyRefs.current[name] = el; }}
              position={initPos}
            >
              <mesh>
                <sphereGeometry args={[radius, 32, 32]} />
                <meshStandardMaterial
                  color={color}
                  emissive={isEmissive ? color : '#000000'}
                  emissiveIntensity={isEmissive ? 0.6 : 0}
                  roughness={0.4}
                  metalness={0.1}
                />
              </mesh>
              {/* Glow ring */}
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <ringGeometry args={[radius * 1.2, radius * 1.5, 32]} />
                <meshBasicMaterial color={color} transparent opacity={0.15} side={THREE.DoubleSide} />
              </mesh>
              {isFullMode && (
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

            {/* Trail line — only re-renders when currentFrame changes via parent */}
            {visibleTrails[name] && visibleTrails[name].length > 1 && (
              <Line
                points={visibleTrails[name]}
                color={color}
                lineWidth={1.2}
                transparent
                opacity={0.5}
              />
            )}
          </React.Fragment>
        );
      })}
    </group>
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
            {s}{'\u00D7'}
          </button>
        ))}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// Main OrbitVisualizer
//
// Architecture:
//   - ONE <Canvas> that is ALWAYS mounted (never unmounts → no context loss)
//   - The wrapper div toggles CSS class for mini ↔ full layout
//   - Body positions updated via Three.js refs in useFrame (no React rerenders)
//   - Trail geometry updated only when currentFrame changes (parent-driven)
//   - Stars use FIXED props — no dynamic count/radius to avoid geometry errors
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

  useEffect(() => {
    if (visible) fetchTrajectory();
  }, [visible, fetchTrajectory]);

  // Playback loop — updates currentFrame state at ~30fps
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

  // Escape key
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

  // Collapse when hidden
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

  return (
    <>
      {/* Backdrop overlay — full mode only */}
      {isFullMode && (
        <div className="viz-backdrop" onClick={() => setIsFullMode(false)} />
      )}

      {/* Single persistent panel — class toggles mini/full */}
      <div className={isFullMode ? 'viz-panel viz-panel--full' : 'viz-panel viz-panel--mini'}>

        {/* Header — full mode only */}
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

        {/* Canvas area */}
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
              <directionalLight position={[20, 30, 10]} intensity={0.3} />

              {/* Stars — FIXED props, never change after mount */}
              <Stars radius={100} depth={50} count={1500} factor={4} saturation={0.3} fade speed={0.4} />

              {/* Camera behaviour */}
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

              {/* Scene content — bodies + trails */}
              <SceneContent
                snapshots={trajectoryData.snapshots}
                meta={trajectoryData.meta}
                isFullMode={isFullMode}
                playbackRef={playbackRef}
                currentFrame={currentFrame}
              />
            </Canvas>
          )}
        </div>

        {/* Mini overlay — click to expand */}
        {!isFullMode && trajectoryData && (
          <div className="viz-mini-overlay" onClick={() => setIsFullMode(true)}>
            <span className="viz-mini-label">{'\u25B6'} ORBIT REPLAY</span>
            <span className="viz-mini-hint">Click to expand</span>
          </div>
        )}

        {/* Playback controls — full mode only */}
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

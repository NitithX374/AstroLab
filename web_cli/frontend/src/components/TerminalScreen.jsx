import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { API_BASE_URL } from '../config';
import 'xterm/css/xterm.css';

const COMMANDS = ['help', 'clear', 'history', 'ask', 'simulate', 'show', 'create', 'demo'];

const BANNER = `\x1b[1;36m
   █████╗ ███████╗████████╗██████╗  ██████╗ ██╗      █████╗ ██████╗ 
  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██║     ██╔══██╗██╔══██╗
  ███████║███████╗   ██║   ██████╔╝██║   ██║██║     ███████║██████╔╝
  ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║██║     ██╔══██║██╔══██╗
  ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝███████╗██║  ██║██████╔╝
  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝ 
\x1b[0m
   \x1b[1;33mAstroLab CLI v3.0\x1b[0m | N-Body & General Relativity Simulation
  \x1b[1;30m---------------------------------------------------------\x1b[0m
   Type \x1b[1;32mhelp\x1b[0m for commands | Type \x1b[1;35mask\x1b[0m for AI Chat
`;

// ─── Helper formatters ────────────────────────────────────────────────────────
function fmtSci(val, unit = '') {
  if (val === null || val === undefined) return '—';
  if (Math.abs(val) === 0) return `0 ${unit}`;
  const exp = Math.floor(Math.log10(Math.abs(val)));
  const coef = val / Math.pow(10, exp);
  if (Math.abs(exp) < 4) return `${val.toFixed(3)} ${unit}`;
  return `${coef.toFixed(3)}×10^${exp} ${unit}`;
}

function fmtTime(s) {
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${(s / 60).toFixed(2)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(2)} hr`;
  if (s < 31536000) return `${(s / 86400).toFixed(2)} day`;
  return `${(s / 31536000).toFixed(3)} yr`;
}

const BODY_TYPE_COLORS = {
  star:         '#FFD166',
  black_hole:   '#EE6055',
  planet:       '#06D6A0',
  moon:         '#8BE8FD',
  asteroid:     '#ADB5BD',
  spacecraft:   '#A78BFA',
};

function bodyTypeColor(type) {
  if (!type) return '#94A3B8';
  const key = type.toLowerCase().replace(' ', '_');
  return BODY_TYPE_COLORS[key] || '#94A3B8';
}

function bodyTypeIcon(type) {
  if (!type) return '⬤';
  const t = type.toLowerCase();
  if (t.includes('star'))       return '★';
  if (t.includes('black'))      return '◉';
  if (t.includes('planet'))     return '●';
  if (t.includes('moon'))       return '◑';
  if (t.includes('asteroid'))   return '◆';
  if (t.includes('spacecraft')) return '▲';
  return '⬤';
}

// ─── Body Card ────────────────────────────────────────────────────────────────
function BodyCard({ body, isExpanded, onToggle }) {
  const color = bodyTypeColor(body.type);
  const icon  = bodyTypeIcon(body.type);

  const rs_km = body.schwarzschild_radius_m / 1000;
  const speedMs = Math.sqrt(
    body.velocity.x ** 2 + body.velocity.y ** 2 + body.velocity.z ** 2
  );

  return (
    <div className="body-card" style={{ '--body-color': color }} onClick={onToggle}>
      <div className="body-card-header">
        <span className="body-icon" style={{ color }}>{icon}</span>
        <div className="body-name-block">
          <span className="body-name">{body.name}</span>
          <span className="body-type-badge" style={{ background: color + '22', color }}>
            {body.type || 'unknown'}
          </span>
        </div>
        <span className="body-expand-chevron">{isExpanded ? '▲' : '▼'}</span>
      </div>

      <div className="body-params">
        <ParamRow label="Mass"               value={fmtSci(body.mass_kg, 'kg')} />
        <ParamRow label="Radius"             value={fmtSci(body.radius_m, 'm')} />
        <ParamRow label="Schwarzschild r"    value={fmtSci(rs_km, 'km')} highlight={rs_km > body.radius_m / 1000} />
        <ParamRow label="Speed"              value={fmtSci(speedMs, 'm/s')} />
      </div>

      {isExpanded && (
        <div className="body-extra">
          <div className="extra-section-label">POSITION (m)</div>
          <div className="xyz-row">
            <span><em>x</em> {fmtSci(body.position.x)}</span>
            <span><em>y</em> {fmtSci(body.position.y)}</span>
            <span><em>z</em> {fmtSci(body.position.z)}</span>
          </div>
          <div className="extra-section-label" style={{ marginTop: '8px' }}>VELOCITY (m/s)</div>
          <div className="xyz-row">
            <span><em>x</em> {fmtSci(body.velocity.x)}</span>
            <span><em>y</em> {fmtSci(body.velocity.y)}</span>
            <span><em>z</em> {fmtSci(body.velocity.z)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function ParamRow({ label, value, highlight }) {
  return (
    <div className="param-row">
      <span className="param-label">{label}</span>
      <span className={`param-value ${highlight ? 'param-highlight' : ''}`}>{value}</span>
    </div>
  );
}

// ─── Right Panel ──────────────────────────────────────────────────────────────
function InspectorPanel({ state, loading }) {
  const [expanded, setExpanded] = useState({});

  const toggle = (name) => setExpanded(prev => ({ ...prev, [name]: !prev[name] }));

  if (loading) {
    return (
      <div className="inspector-empty">
        <div className="inspector-spinner" />
        <span>Loading state…</span>
      </div>
    );
  }

  const hasData = state && (state.bodies?.length > 0 || state.bh_config);

  return (
    <div className="inspector-content">
      {/* Sim clock */}
      {state && (
        <div className="sim-clock">
          <div className="sim-clock-row">
            <span className="sim-clock-label">SIM TIME</span>
            <span className="sim-clock-value">{fmtTime(state.sim_time_s || 0)}</span>
          </div>
          <div className="sim-clock-row">
            <span className="sim-clock-label">STEP</span>
            <span className="sim-clock-value">{state.sim_step ?? 0}</span>
          </div>
          <div className="sim-clock-row">
            <span className="sim-clock-label">Δt</span>
            <span className="sim-clock-value">{fmtTime(state.sim_dt_s || 60)}</span>
          </div>
        </div>
      )}

      {/* GR / Black Hole Banner */}
      {state?.bh_config && (
        <div className="bh-banner">
          <div className="bh-banner-title">
            <span style={{ color: '#EE6055' }}>◉</span> BLACK HOLE (GR)
          </div>
          <ParamRow label="Metric"   value={state.bh_config.metric} />
          <ParamRow label="Spin"     value={state.bh_config.spin?.toFixed(4) ?? '0'} />
          <ParamRow label="Mass"     value={fmtSci(state.bh_config.mass_kg, 'kg')} />
          <ParamRow label="rₛ"       value={fmtSci(state.bh_config.schwarzschild_radius_km, 'km')} />
        </div>
      )}

      {/* Body list */}
      {state?.bodies?.length > 0 ? (
        <>
          <div className="inspector-section-label">
            BODIES &nbsp;<span className="body-count">{state.bodies.length}</span>
          </div>
          {state.bodies.map(b => (
            <BodyCard
              key={b.name}
              body={b}
              isExpanded={!!expanded[b.name]}
              onToggle={() => toggle(b.name)}
            />
          ))}
        </>
      ) : !hasData ? (
        <div className="inspector-empty">
          <div className="inspector-empty-icon">🌌</div>
          <p className="inspector-empty-title">No bodies in simulation</p>
          <p className="inspector-empty-hint">
            Try: <code>create body Sun star</code>
          </p>
        </div>
      ) : null}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function TerminalScreen({ onLogout }) {
  const terminalRef   = useRef(null);
  const termInstance  = useRef(null);
  const fitAddonInstance = useRef(null);
  const inputBuffer   = useRef('');
  const history       = useRef([]);
  const historyIndex  = useRef(-1);
  const isStreaming   = useRef(false);
  const isQueryMode   = useRef(false);
  const chatHistory   = useRef([]);

  const [simState, setSimState]     = useState(null);
  const [stateLoading, setStateLoading] = useState(false);

  // Fetch simulation state from backend
  const refreshState = useCallback(async () => {
    setStateLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/state`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setSimState(data);
      }
    } catch (_) {}
    setStateLoading(false);
  }, []);

  useEffect(() => {
    refreshState(); // initial load
  }, [refreshState]);

  useEffect(() => {
    const term = new Terminal({
      cursorBlink: true,
      scrollback: 5000,
      theme: {
        background: '#0B0E14',
        foreground: '#E2E8F0',
        cursor: '#3A86FF',
        black: '#000000',
        red: '#FF006E',
        green: '#38B000',
        yellow: '#f1fa8c',
        blue: '#3A86FF',
        magenta: '#8338EC',
        cyan: '#8be9fd',
        white: '#ffffff',
      },
      fontFamily: 'JetBrains Mono, monospace',
      fontSize: 14,
    });

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.open(terminalRef.current);
    fitAddon.fit();

    termInstance.current    = term;
    fitAddonInstance.current = fitAddon;

    term.write(BANNER.replace(/\n/g, '\r\n'));
    term.writeln('');
    promptUser();

    const ro = new ResizeObserver(() => {
      try { fitAddon.fit(); } catch (_) {}
    });
    ro.observe(terminalRef.current);

    const resizeListener = () => { try { fitAddon.fit(); } catch (_) {} };
    window.addEventListener('resize', resizeListener);

    term.onKey(({ key, domEvent }) => {
      if (isStreaming.current) return;
      const printable = !domEvent.altKey && !domEvent.ctrlKey && !domEvent.metaKey;

      if (domEvent.keyCode === 13) {
        term.write('\r\n');
        handleCommand(inputBuffer.current.trim());
      } else if (domEvent.keyCode === 8) {
        if (inputBuffer.current.length > 0) {
          inputBuffer.current = inputBuffer.current.slice(0, -1);
          term.write('\b \b');
        }
      } else if (domEvent.keyCode === 38) {
        if (history.current.length > 0) {
          if (historyIndex.current < history.current.length - 1) {
            historyIndex.current++;
            replaceInput(history.current[history.current.length - 1 - historyIndex.current]);
          }
        }
      } else if (domEvent.keyCode === 40) {
        if (historyIndex.current >= 0) {
          historyIndex.current--;
          if (historyIndex.current === -1) {
            replaceInput('');
          } else {
            replaceInput(history.current[history.current.length - 1 - historyIndex.current]);
          }
        }
      } else if (domEvent.keyCode === 9) {
        if (!isQueryMode.current) {
          const partial = inputBuffer.current;
          const match = COMMANDS.find(cmd => cmd.startsWith(partial));
          if (match) replaceInput(match);
        }
      } else if (printable) {
        inputBuffer.current += key;
        term.write(key);
      }
    });

    return () => {
      window.removeEventListener('resize', resizeListener);
      ro.disconnect();
      term.dispose();
    };
  }, []);

  const replaceInput = (newInput) => {
    const term = termInstance.current;
    for (let i = 0; i < inputBuffer.current.length; i++) term.write('\b \b');
    inputBuffer.current = newInput;
    term.write(newInput);
  };

  const promptUser = () => {
    inputBuffer.current = '';
    historyIndex.current = -1;
    if (isQueryMode.current) {
      termInstance.current.write('\x1b[1;35mquery>\x1b[0m ');
    } else {
      termInstance.current.write('\x1b[1;32mastrolab>\x1b[0m ');
    }
  };

  const handleCommand = async (rawCmd) => {
    const term = termInstance.current;
    const cmd = rawCmd.trim();
    const cmdLower = cmd.toLowerCase();

    if (!cmd) { promptUser(); return; }
    history.current.push(cmd);

    if (isQueryMode.current) {
      if (cmdLower === 'exit' || cmdLower === 'quit') {
        term.writeln('\x1b[1;33mexiting LLM query mode…\x1b[0m');
        isQueryMode.current = false;
        promptUser();
        return;
      }
      if (cmdLower === 'clear') { term.clear(); promptUser(); return; }
      await handleAskStream(cmd);
      return;
    }

    if (cmdLower === 'clear') { term.clear(); promptUser(); return; }

    if (cmdLower === 'ask') {
      term.writeln('\x1b[1;35m>>> entering AI Chat mode (type "exit" to leave)\x1b[0m');
      isQueryMode.current = true;
      promptUser();
      return;
    }

    if (cmdLower.startsWith('ask ')) {
      const promptText = cmd.slice(4).trim();
      if (!promptText) {
        term.writeln('\x1b[1;31mError:\x1b[0m Please provide a prompt.');
        promptUser();
        return;
      }
      await handleAskStream(promptText);
      return;
    }

    if (cmdLower === 'help') {
      term.writeln('\x1b[1;36mWEB CLI SPECIFIC COMMANDS:\x1b[0m');
      term.writeln('  \x1b[1;33mclear\x1b[0m     - Clear terminal output');
      term.writeln('  \x1b[1;33mhistory\x1b[0m   - Show command history');
      term.writeln('  \x1b[1;33mask\x1b[0m       - Enter persistent AI Chat mode');
      term.writeln('  \x1b[1;33mask <q>\x1b[0m   - Ask a single AI question');
      term.writeln('');
      term.writeln('\x1b[1;36mASTROLAB PHYSICS COMMANDS (Powered by Python Engine):\x1b[0m');
      await handleExecutionStream('help');
      return;
    }

    if (cmdLower === 'history') {
      history.current.forEach((h, i) => term.writeln(`  ${i + 1}  ${h}`));
      promptUser();
      return;
    }

    await handleExecutionStream(cmd);
  };

  const handleExecutionStream = async (command) => {
    isStreaming.current = true;
    const term = termInstance.current;

    try {
      const response = await fetch(`${API_BASE_URL}/execute/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
        credentials: 'include',
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const reader  = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '{"text": "[DONE]"}') break;
            try {
              const payload = JSON.parse(dataStr);
              if (payload.text) {
                const formattedText = payload.text.replace(/(?<!\r)\n/g, '\r\n');
                term.write(formattedText);
                const vizMatch = payload.text.match(/Plotly orbit chart saved → '([^']+)'/);
                if (vizMatch) {
                  setTimeout(() => window.open(`${API_BASE_URL}/outputs/${vizMatch[1]}`, '_blank'), 500);
                }
              }
            } catch (_) {}
          }
        }
      }
    } catch (error) {
      term.writeln(`\r\n\x1b[1;31mServer Error:\x1b[0m ${error.message}`);
    }

    isStreaming.current = false;
    promptUser();
    // Refresh the right panel after every engine command
    refreshState();
  };

  const handleAskStream = async (prompt) => {
    isStreaming.current = true;
    const term = termInstance.current;

    chatHistory.current.push({ role: 'user', content: prompt });
    if (chatHistory.current.length > 10) chatHistory.current.shift();

    term.write('\x1b[1;34mSystem:\x1b[0m ');
    let fullResponse = '';

    try {
      const response = await fetch(`${API_BASE_URL}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, messages: chatHistory.current }),
        credentials: 'include',
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const reader  = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') break;
            else if (dataStr.startsWith('[ERROR]')) {
              term.write(`\r\n\x1b[1;31mSystem Error:\x1b[0m ${dataStr.slice(7)}\r\n`);
              break;
            } else {
              try {
                const payload = JSON.parse(dataStr);
                if (payload.text) {
                  const formattedText = payload.text.replace(/(?<!\r)\n/g, '\r\n');
                  term.write(formattedText);
                  fullResponse += payload.text;
                }
              } catch (_) {}
            }
          }
        }
      }

      if (fullResponse) chatHistory.current.push({ role: 'assistant', content: fullResponse });
      term.writeln('');
    } catch (error) {
      term.writeln(`\r\n\x1b[1;31mNetwork Error:\x1b[0m ${error.message}`);
    }

    isStreaming.current = false;
    promptUser();
  };

  return (
    <div className="lab-layout">
      {/* ── Left: Terminal ── */}
      <div className="terminal-wrapper">
        <div className="terminal-header">
          <div className="mac-controls">
            <div className="mac-dot red" />
            <div className="mac-dot yellow" />
            <div className="mac-dot green" />
          </div>
          <div className="terminal-title">TERMINAL // ASTROLAB</div>
          <button className="terminal-logout" onClick={onLogout}>DISCONNECT</button>
        </div>
        <div className="terminal-container" ref={terminalRef} />
      </div>

      {/* ── Right: Inspector ── */}
      <div className="inspector-panel">
        <div className="inspector-header">
          <span className="inspector-header-icon">⚛</span>
          <span className="inspector-header-title">BODY INSPECTOR</span>
          <button className="inspector-refresh-btn" onClick={refreshState} title="Refresh state">↻</button>
        </div>
        <div className="inspector-scroll">
          <InspectorPanel state={simState} loading={stateLoading} />
        </div>
      </div>
    </div>
  );
}

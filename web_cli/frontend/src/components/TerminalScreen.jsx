import React, { useEffect, useRef, useState } from 'react';
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
   Type \x1b[1;32mhelp\x1b[0m to list commands
   Type \x1b[1;34mask <question>\x1b[0m to talk to AI
`;

export default function TerminalScreen({ onLogout }) {
  const terminalRef = useRef(null);
  const termInstance = useRef(null);
  const fitAddonInstance = useRef(null);
  const inputBuffer = useRef('');
  const history = useRef([]);
  const historyIndex = useRef(-1);
  const isStreaming = useRef(false);

  useEffect(() => {
    // Initialize xterm
    const term = new Terminal({
      cursorBlink: true,
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
    
    termInstance.current = term;
    fitAddonInstance.current = fitAddon;

    // xterm.js strictly requires \r\n for line breaks. Template literals use \n.
    term.write(BANNER.replace(/\n/g, '\r\n'));
    term.writeln('');
    promptUser();

    // Handle Resize
    const resizeListener = () => fitAddon.fit();
    window.addEventListener('resize', resizeListener);

    // Keystroke handler
    term.onKey(({ key, domEvent }) => {
      if (isStreaming.current) return; // Block input while streaming
      
      const printable = !domEvent.altKey && !domEvent.ctrlKey && !domEvent.metaKey;
      
      if (domEvent.keyCode === 13) { // Enter
        term.write('\r\n');
        handleCommand(inputBuffer.current.trim());
      } else if (domEvent.keyCode === 8) { // Backspace
        if (inputBuffer.current.length > 0) {
          inputBuffer.current = inputBuffer.current.slice(0, -1);
          term.write('\b \b');
        }
      } else if (domEvent.keyCode === 38) { // Up Arrow
        if (history.current.length > 0) {
          if (historyIndex.current < history.current.length - 1) {
            historyIndex.current++;
            replaceInput(history.current[history.current.length - 1 - historyIndex.current]);
          }
        }
      } else if (domEvent.keyCode === 40) { // Down Arrow
        if (historyIndex.current >= 0) {
          historyIndex.current--;
          if (historyIndex.current === -1) {
            replaceInput('');
          } else {
            replaceInput(history.current[history.current.length - 1 - historyIndex.current]);
          }
        }
      } else if (domEvent.keyCode === 9) { // Tab (Autocomplete)
        const partial = inputBuffer.current;
        const match = COMMANDS.find(cmd => cmd.startsWith(partial));
        if (match) {
          replaceInput(match);
        }
      } else if (printable) {
        inputBuffer.current += key;
        term.write(key);
      }
    });

    return () => {
      window.removeEventListener('resize', resizeListener);
      term.dispose();
    };
  }, []);

  const replaceInput = (newInput) => {
    const term = termInstance.current;
    // Clear current input from screen
    for (let i = 0; i < inputBuffer.current.length; i++) {
        term.write('\b \b');
    }
    inputBuffer.current = newInput;
    term.write(newInput);
  };

  const promptUser = () => {
    inputBuffer.current = '';
    historyIndex.current = -1;
    termInstance.current.write('\x1b[1;32mastrolab>\x1b[0m ');
  };

  const handleCommand = async (cmd) => {
    const term = termInstance.current;
    
    if (!cmd) {
      promptUser();
      return;
    }

    history.current.push(cmd);

    if (cmd === 'clear') {
      term.clear();
      promptUser();
      return;
    }

    if (cmd === 'help') {
      term.writeln('\x1b[1;36mWEB CLI SPECIFIC COMMANDS:\x1b[0m');
      term.writeln('  \x1b[1;33mclear\x1b[0m     - Clear terminal output');
      term.writeln('  \x1b[1;33mhistory\x1b[0m   - Show command history');
      term.writeln('  \x1b[1;33mask\x1b[0m       - Ask the AI (e.g. ask <prompt>)');
      term.writeln('');
      term.writeln('\x1b[1;36mASTROLAB PHYSICS COMMANDS (Powered by Python Engine):\x1b[0m');
      
      // Pass the 'help' command to the actual Python engine backend
      await handleExecutionStream('help');
      return;
    }

    if (cmd === 'history') {
      history.current.forEach((h, i) => term.writeln(`  ${i+1}  ${h}`));
      promptUser();
      return;
    }

    if (cmd.startsWith('ask ')) {
      const promptText = cmd.slice(4).trim();
      if (!promptText) {
        term.writeln('\x1b[1;31mError:\x1b[0m Please provide a prompt. Usage: ask <prompt>');
        promptUser();
        return;
      }
      
      await handleAskStream(promptText);
      return;
    }

    // Instead of Command Not Found, assume it's an AstroLab command
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
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '{"text": "[DONE]"}') {
              break; // End of stream
            }
            try {
                const payload = JSON.parse(dataStr);
                if (payload.text) {
                    // xterm.js requires \r\n to return to the beginning of the next line.
                    // Python print() usually just outputs \n.
                    const formattedText = payload.text.replace(/(?<!\r)\n/g, '\r\n');
                    term.write(formattedText);
                    
                    // Hook for opening visualization output on web
                    const vizMatch = payload.text.match(/Plotly orbit chart saved → '([^']+)'/);
                    if (vizMatch) {
                      const filename = vizMatch[1];
                      setTimeout(() => {
                        window.open(`${API_BASE_URL}/outputs/${filename}`, '_blank');
                      }, 500);
                    }
                }
            } catch (e) {
                // Ignore incomplete JSON chunks from splitting
            }
          }
        }
      }
    } catch (error) {
      term.writeln(`\r\n\x1b[1;31mServer Error:\x1b[0m ${error.message}`);
    }
    
    isStreaming.current = false;
    promptUser();
  };

  const handleAskStream = async (prompt) => {
    isStreaming.current = true;
    const term = termInstance.current;
    
    term.write('\x1b[1;34mSystem:\x1b[0m ');

    try {
      const response = await fetch(`${API_BASE_URL}/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        
        // SSE parsing
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') {
              // End of stream
              break;
            } else if (dataStr.startsWith('[ERROR]')) {
              term.write(`\r\n\x1b[1;31mSystem Error:\x1b[0m ${dataStr.slice(7)}\r\n`);
              break;
            } else {
              // Write token
              try {
                const payload = JSON.parse(dataStr);
                if (payload.text) {
                  const formattedText = payload.text.replace(/(?<!\r)\n/g, '\r\n');
                  term.write(formattedText);
                }
              } catch (e) {
                // Ignore incomplete
              }
            }
          }
        }
      }
      term.writeln(''); // newline after response
    } catch (error) {
      term.writeln(`\r\n\x1b[1;31mNetwork Error:\x1b[0m ${error.message}`);
    }
    
    isStreaming.current = false;
    promptUser();
  };

  return (
    <div className="terminal-wrapper">
      <div className="terminal-header">
        <div className="mac-controls">
          <div className="mac-dot red"></div>
          <div className="mac-dot yellow"></div>
          <div className="mac-dot green"></div>
        </div>
        <div className="terminal-title">TERMINAL // ASTROLAB</div>
        <button className="terminal-logout" onClick={onLogout}>DISCONNECT</button>
      </div>
      <div className="terminal-container" ref={terminalRef}></div>
    </div>
  );
}

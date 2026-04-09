import asyncio
import json
import threading
import queue
import time
from typing import Dict, Any, List

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

class SimulationStreamer:
    """
    Bridge that ingests synchronous physics frames from a high-speed Python loop
    and emits them at a fixed broadcast frequency (e.g. 60 Hz) to connected WebSocket clients (UE5).
    """

    def __init__(self, port: int = 12200, broadcast_hz: float = 60.0, buffer_size: int = 100):
        self.port = port
        self.broadcast_hz = broadcast_hz
        self.delay = 1.0 / self.broadcast_hz
        
        # Frame Buffer defined in architecture
        self._frame_queue = queue.Queue(maxsize=buffer_size)
        
        self._loop = None
        self._thread = None
        self._running = False
        self.clients = set()

    def start(self):
        if not HAS_WEBSOCKETS:
            print("  [!] Cannot start Network Bridge: 'websockets' library is missing.")
            return False
            
        self._running = True
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)

    def enqueue_frame(self, frame_id: int, time_jd: float, bodies: List[Any]):
        """
        Called by the ultra-fast synchronous physics loop. Pushes state to safe queue.
        Overrides old frames to ensure downsampling if physics > broadcast tick.
        """
        if not self._running:
            return
            
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        # Build strictly MKS UE5 JSON schema
        json_objs = []
        for b in bodies:
            obj_data = {
                "id": getattr(b, 'name', 'unknown'),
                "type": getattr(b, 'obj_type', 'planet'),
                "pos": [b.position.x, b.position.y, b.position.z],
                "vel": [b.velocity.x, b.velocity.y, b.velocity.z],
                "radius": getattr(b, 'radius', 1.0),
                "mass": getattr(b, 'mass', 1.0),
                "color": getattr(b, 'color', 'white'),
            }
            # Catch dynamic black hole properties if simulated via GR engine
            if hasattr(b, 'spin'):
                obj_data['spin'] = b.spin
                
            json_objs.append(obj_data)
            
        data = {
            "frame_id": frame_id,
            "time_jd": time_jd,
            "objects": json_objs
        }
        
        try:
            self._frame_queue.put_nowait(data)
        except queue.Full:
            pass

    def _run_async_loop(self):
        """Daemon thread running the Async network loops."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        start_server = websockets.serve(self._ws_handler, "0.0.0.0", self.port)
        self._loop.run_until_complete(start_server)
        
        # Attach 60hz clock task
        self._loop.create_task(self._broadcast_task())
        
        print(f"\n  [Network] AstroLab WebSocket Bridge online (ws://0.0.0.0:{self.port})")
        print(f"  [Network] Downsampling Physics to {self.broadcast_hz} Hz Broadcasting...")
        self._loop.run_forever()

    async def _ws_handler(self, websocket, path):
        """Registers a new UE5 client."""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def _broadcast_task(self):
        """The Broadcast Clock Loop"""
        while self._running:
            start_t = time.time()
            
            if self.clients:
                try:
                    # Get the most recent frame
                    data = self._frame_queue.get_nowait()
                    
                    # Flush queue so UE5 receives strictly the last calculated physics tick
                    while not self._frame_queue.empty():
                        data = self._frame_queue.get_nowait()
                        
                    msg = json.dumps(data)
                    
                    # Fanout
                    tasks = [asyncio.create_task(client.send(msg)) for client in self.clients]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                        
                except queue.Empty:
                    pass
            
            # Sync to Broadcast Clock
            elapsed = time.time() - start_t
            remain = self.delay - elapsed
            if remain > 0:
                await asyncio.sleep(remain)
            else:
                await asyncio.sleep(0)

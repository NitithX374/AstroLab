import sys
import os

# Fix UTF-8 encoding for Windows console (if needed for printing)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, '.')
from astrolab.cli.parser import AstroLabCLI

def test_plotly_render():
    print("\n--- Testing Plotly HTML Rendering ---")
    cli = AstroLabCLI()
    
    # Setup a simple system
    cli.onecmd("create body sun mass=1.989e30 radius=6.96e8 type=star color=yellow")
    cli.onecmd("create body earth mass=5.97e24 pos=(1AU,0,0) vel=(0,29.78km/s,0) radius=6.37e6 type=planet color=blue")
    
    # Simulate with visualize=on
    output_html = "test_orbits.html"
    if os.path.exists(output_html):
        os.remove(output_html)
        
    print("Running simulation (visualize=on output=test_orbits.html)...")
    cli.onecmd(f"simulate dt=3600 steps=100 visualize=on output={output_html}")
    
    if os.path.exists(output_html):
        print(f"[SUCCESS] Plotly HTML rendered: {output_html} ({os.path.getsize(output_html)} bytes)")
    else:
        print("[FAILURE] Plotly HTML was NOT rendered.")

def test_vispy_import():
    print("\n--- Testing Vispy Imports & Recorder Setup ---")
    try:
        from astrolab.viz.recorder import TrajectoryRecorder
        from astrolab.viz.vispy_viz import AstroLabReplay
        print("[SUCCESS] Vispy modules imported correctly.")
        
        # Test recorder save/load
        r = TrajectoryRecorder()
        r._meta['test'] = {'color': '#FF0000', 'type': 'star', 'mass': 1.0, 'radius': 1.0}
        r._snapshots.append({'time': 0, 'bodies': [{'name': 'test', 'x': 1, 'y': 2, 'z': 3, 'speed': 0}]})
        
        test_json = "test_traj.json"
        r.save(test_json)
        r2 = TrajectoryRecorder.load(test_json)
        
        if r2.get_body_names() == ['test']:
            print("[SUCCESS] TrajectoryRecorder save/load works.")
        else:
            print("[FAILURE] TrajectoryRecorder data mismatch.")
        os.remove(test_json)
        
    except ImportError as e:
        print(f"[FAILURE] Vispy import failed: {e}")
    except Exception as e:
        print(f"[ERROR] recorder test failed: {e}")

if __name__ == "__main__":
    test_plotly_render()
    test_vispy_import()
    print("\nSmoke tests complete.")

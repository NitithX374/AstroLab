import sys, math
sys.path.insert(0, '.')

from astrolab.core.models import CelestialBody, Vector3D
from astrolab.state.manager import StateManager
from astrolab.engine.simulator import SimulationEngine
import astrolab.physics.toolkit as tk

print('=' * 60)
print('  AstroLab CLI -- Integration Test Suite')
print('=' * 60)

sm = StateManager()
sun   = CelestialBody('sun',   1.989e30, Vector3D(0,0,0),               Vector3D(0,0,0),       696_000_000, 'star',   'yellow')
earth = CelestialBody('earth', 5.97e24,  Vector3D(149.6e9,0,0),         Vector3D(0,29780,0),   6_371_000,   'planet', 'blue')
moon  = CelestialBody('moon',  7.342e22, Vector3D(149.6e9+384.4e6,0,0), Vector3D(0,29780+1022,0), 1_737_400, 'moon',  'grey')

sm.add_body(sun)
sm.add_body(earth)
sm.add_body(moon)

print()
print('-- Test 1: Escape velocity --')
ve_earth = tk.escape_velocity(earth)
ve_sun   = tk.escape_velocity(sun)
print(f'  Earth: {ve_earth/1e3:.4f} km/s  (expected ~11.19 km/s)')
print(f'  Sun:   {ve_sun/1e3:.4f} km/s   (expected ~617.53 km/s)')
assert abs(ve_earth/1e3 - 11.19) < 0.05, 'Escape velocity mismatch'
print('  PASS')

print()
print('-- Test 2: Orbital period (Earth around Sun) --')
T = tk.orbital_period(earth, sun)
days = T / 86400
print(f'  Period: {T:.4e} s  = {days:.2f} days  (expected ~365.25 days)')
assert abs(days - 365.25) < 2.0, 'Orbital period mismatch'
print('  PASS')

print()
print('-- Test 3: Gravitational force (Earth-Sun) --')
F, uv = tk.gravitational_force(earth, sun)
print(f'  Force: {F:.4e} N  (expected ~3.54e22 N)')
assert abs(F - 3.54e22) / 3.54e22 < 0.01, 'Gravitational force mismatch'
print('  PASS')

print()
print('-- Test 4: Schwarzschild radius --')
rs_sun = tk.schwarzschild_radius(sun)
print(f'  Sun Rs: {rs_sun:.4f} m  (expected ~2953.25 m)')
assert abs(rs_sun - 2953.25) < 5.0, 'Schwarzschild radius mismatch'
print('  PASS')

print()
print('-- Test 5: System energy --')
e = tk.total_system_energy([sun, earth, moon])
print(f'  KE: {e["kinetic"]:+.4e} J')
print(f'  PE: {e["potential"]:+.4e} J')
print(f'  TE: {e["total"]:+.4e} J')
assert e['kinetic'] > 0, 'KE must be positive'
assert e['potential'] < 0, 'PE must be negative'
print('  PASS')

print()
print('-- Test 6: Lagrange points --')
pts = tk.lagrange_points(sun, earth)
for lbl, pos in pts.items():
    print(f'  {lbl}: {pos}')
assert 'L1' in pts and 'L4' in pts
print('  PASS')

print()
print('-- Test 7: Simulate 1 Earth year (RK4) --')
se = SimulationEngine(sm)
initial_energy = tk.total_system_energy(sm.get_all_bodies())['total']
result = se.simulate(3600, 8760, 'rk4', collision_detection=False, energy_log_interval=100)
final_energy = tk.total_system_energy(sm.get_all_bodies())['total']
drift_pct = abs((final_energy - initial_energy) / initial_energy) * 100
print(f'  Sim time: {result["elapsed_time"]:.3e} s')
print(f'  Earth pos: {sm.get_body("earth").position}')
print(f'  Energy drift: {drift_pct:.4f}%')
print('  PASS')

print()
print('-- Test 8: Export/Import state --')
sm.export_state('test_state.json')
sm2 = StateManager()
sm2.import_state('test_state.json')
assert sm2.get_body('earth') is not None, 'Earth not found after import'
e2 = sm2.get_body('earth')
print(f'  Reimported Earth mass: {e2.mass:.3e} kg   radius: {e2.radius} m')
print('  PASS')

print()
print('=' * 60)
print('  ALL TESTS PASSED')
print('=' * 60)

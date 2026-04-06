"""
Test Suite — General Relativity Physics
========================================
Comprehensive tests for the GR black hole simulation module.

Tests
-----
1. Schwarzschild metric symmetry and determinant
2. Christoffel symbol spot-checks (analytic vs. computed)
3. Circular orbit stability at r=10M
4. Photon deflection in weak-field limit
5. Horizon absorption (radial infall)
6. Kerr → Schwarzschild reduction at a=0
7. Redshift formula verification
8. Constants of motion conservation (E, L)
9. Time dilation at known radii
10. ISCO and photon sphere radii
"""

import math
import sys
import os

# Windows: reconfigure stdout to UTF-8 for Unicode output
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_schwarzschild_metric_properties():
    """Test that the Schwarzschild metric is diagonal and has correct determinant."""
    from astrolab.physics.metrics import SchwarzschildMetric

    M = 1.0  # geometric units
    metric = SchwarzschildMetric(M)

    # At r=10M, θ=π/2
    r, theta = 10.0, math.pi / 2
    g = metric.metric_tensor(r, theta)

    # Should be diagonal
    for i in range(4):
        for j in range(4):
            if i != j:
                assert abs(g[i, j]) < 1e-15, f"Off-diagonal g[{i},{j}] = {g[i,j]}"

    # Check specific values
    f = 1.0 - 2.0 * M / r  # 0.8
    assert abs(g[0, 0] - (-f)) < 1e-12, f"g_tt = {g[0,0]}, expected {-f}"
    assert abs(g[1, 1] - (1.0 / f)) < 1e-12, f"g_rr = {g[1,1]}, expected {1/f}"
    assert abs(g[2, 2] - r * r) < 1e-12, f"g_θθ = {g[2,2]}, expected {r*r}"
    assert abs(g[3, 3] - r * r) < 1e-12, f"g_φφ = {g[3,3]}, expected {r*r}"  # sin(π/2) = 1

    # Determinant: det(g) = -r^4 sin^2(θ)
    det = np.linalg.det(g)
    expected_det = -(r ** 4) * (math.sin(theta) ** 2)
    assert abs(det - expected_det) < 1e-6, f"det(g) = {det}, expected {expected_det}"

    # Inverse: g * g^-1 = I
    gi = metric.inverse_metric(r, theta)
    identity = g @ gi
    for i in range(4):
        for j in range(4):
            expected = 1.0 if i == j else 0.0
            assert abs(identity[i, j] - expected) < 1e-10, \
                f"(g·g^-1)[{i},{j}] = {identity[i,j]}, expected {expected}"

    print("  ✓ Schwarzschild metric: diagonal, correct values, det = -r⁴sin²θ, g·g⁻¹ = I")


def test_schwarzschild_christoffel():
    """Spot-check Schwarzschild Christoffel symbols against analytic formulas."""
    from astrolab.physics.metrics import SchwarzschildMetric

    M = 1.0
    metric = SchwarzschildMetric(M)
    r, theta = 10.0, math.pi / 3
    Gamma = metric.christoffel(r, theta)

    rm2M = r - 2.0 * M

    # Γ^t_{tr} = M / (r(r-2M))
    expected = M / (r * rm2M)
    assert abs(Gamma[0, 0, 1] - expected) < 1e-12, \
        f"Γ^t_tr = {Gamma[0,0,1]}, expected {expected}"

    # Γ^r_{tt} = M(r-2M) / r³
    expected = M * rm2M / (r ** 3)
    assert abs(Gamma[1, 0, 0] - expected) < 1e-12, \
        f"Γ^r_tt = {Gamma[1,0,0]}, expected {expected}"

    # Γ^θ_{rθ} = 1/r
    expected = 1.0 / r
    assert abs(Gamma[2, 1, 2] - expected) < 1e-12, \
        f"Γ^θ_rθ = {Gamma[2,1,2]}, expected {expected}"

    # Γ^φ_{θφ} = cos(θ)/sin(θ)
    expected = math.cos(theta) / math.sin(theta)
    assert abs(Gamma[3, 2, 3] - expected) < 1e-12, \
        f"Γ^φ_θφ = {Gamma[3,2,3]}, expected {expected}"

    print("  ✓ Schwarzschild Christoffel symbols: all spot-checks pass")


def test_horizon_and_radii():
    """Test event horizon, photon sphere, and ISCO radii."""
    from astrolab.physics.metrics import SchwarzschildMetric, KerrMetric

    M = 1.0

    # Schwarzschild
    s = SchwarzschildMetric(M)
    assert abs(s.event_horizon()[0] - 2.0) < 1e-12
    assert abs(s.photon_sphere() - 3.0) < 1e-12
    assert abs(s.isco() - 6.0) < 1e-12

    # Kerr a/M = 0.5
    k = KerrMetric(M, a_over_M=0.5)
    r_plus, r_minus = k.event_horizon()
    assert r_plus > r_minus
    assert abs(r_plus - (1.0 + math.sqrt(1.0 - 0.25))) < 1e-10
    assert abs(r_minus - (1.0 - math.sqrt(1.0 - 0.25))) < 1e-10

    # ISCO: prograde < retrograde for spinning BH
    r_pro = k.isco(prograde=True)
    r_retro = k.isco(prograde=False)
    assert r_pro < r_retro, f"Prograde ISCO ({r_pro}) should be < retrograde ({r_retro})"
    assert r_pro < 6.0, f"Prograde ISCO ({r_pro}) should be < 6M for spinning BH"

    print("  ✓ Horizon, photon sphere, ISCO: all radii correct")


def test_kerr_reduces_to_schwarzschild():
    """Kerr metric with a=0 should produce the same results as Schwarzschild."""
    from astrolab.physics.metrics import SchwarzschildMetric, KerrMetric

    M = 1.0
    s = SchwarzschildMetric(M)
    k = KerrMetric(M, a_over_M=0.0)

    r, theta = 8.0, math.pi / 4

    gs = s.metric_tensor(r, theta)
    gk = k.metric_tensor(r, theta)

    for i in range(4):
        for j in range(4):
            assert abs(gs[i, j] - gk[i, j]) < 1e-8, \
                f"g[{i},{j}]: Schwarzschild={gs[i,j]}, Kerr(a=0)={gk[i,j]}"

    # Event horizon
    assert abs(s.event_horizon()[0] - k.event_horizon()[0]) < 1e-10

    print("  ✓ Kerr(a=0) reduces to Schwarzschild metric")


def test_redshift_formula():
    """Verify gravitational redshift at specific radii."""
    from astrolab.physics.observables import redshift_schwarzschild

    M = 1.0

    # At r=2.1M (very close to horizon)
    z = redshift_schwarzschild(2.1 * M, 1000.0 * M, M)
    z_analytic = math.sqrt((1.0 - 2.0 / 1000.0) / (1.0 - 2.0 / 2.1)) - 1.0
    assert abs(z - z_analytic) < 1e-8, f"z = {z}, expected {z_analytic}"

    # At large r, z → 0
    z_far = redshift_schwarzschild(500.0 * M, 1000.0 * M, M)
    assert abs(z_far) < 0.005, f"z at large r = {z_far}, expected ≈ 0"

    # Emitter at horizon → infinite redshift
    z_horizon = redshift_schwarzschild(2.0 * M, 1000.0 * M, M)
    assert z_horizon == float('inf'), "Redshift at horizon should be infinite"

    print("  ✓ Gravitational redshift: verified at r=2.1M, large r, and horizon")


def test_time_dilation():
    """Verify time dilation factor at specific radii."""
    from astrolab.physics.observables import time_dilation_schwarzschild

    M = 1.0

    # At r=10M: dτ/dt = √(1 - 2/10) = √0.8
    factor = time_dilation_schwarzschild(10.0, M)
    expected = math.sqrt(0.8)
    assert abs(factor - expected) < 1e-12, f"factor = {factor}, expected {expected}"

    # At horizon: dτ/dt = 0
    factor_h = time_dilation_schwarzschild(2.0, M)
    assert factor_h == 0.0, f"Factor at horizon = {factor_h}, expected 0"

    # At infinity: dτ/dt → 1
    factor_inf = time_dilation_schwarzschild(1e6, M)
    assert abs(factor_inf - 1.0) < 1e-5, f"Factor at infinity = {factor_inf}"

    print("  ✓ Time dilation: correct at r=10M, horizon, and infinity")


def test_geodesic_radial_infall():
    """A particle dropped radially should cross the horizon."""
    from astrolab.physics.metrics import SchwarzschildMetric
    from astrolab.physics.geodesic_integrator import GeodesicIntegrator, ParticleType, TerminationReason
    from astrolab.physics.christoffel import normalize_4velocity

    M = 1.0
    metric = SchwarzschildMetric(M)

    # Radial infall from r=10M with zero angular momentum
    state = np.array([0.0, 10.0, math.pi / 2, 0.0,
                       1.0, -0.1, 0.0, 0.0], dtype=np.float64)
    state = normalize_4velocity(state, metric, -1.0)

    integrator = GeodesicIntegrator(
        metric=metric,
        particle_type=ParticleType.TIMELIKE,
        escape_radius=100.0,
    )

    result = integrator.integrate(state, max_affine=500.0, max_steps=50000)

    assert result.termination_reason == TerminationReason.HORIZON_CROSSED, \
        f"Expected horizon crossing, got: {result.termination_reason}"

    # Final r should be near 2M
    final_r = result.trajectory[-1].r
    assert final_r < 2.5 * M, f"Final r = {final_r}, expected < 2.5M"

    print(f"  ✓ Radial infall: crossed horizon at r={final_r:.4f}M "
          f"({result.steps_taken} steps)")


def test_constants_conservation():
    """E and L should be conserved along a geodesic."""
    from astrolab.physics.metrics import SchwarzschildMetric
    from astrolab.physics.geodesic_integrator import GeodesicIntegrator, ParticleType
    from astrolab.physics.christoffel import normalize_4velocity

    M = 1.0
    metric = SchwarzschildMetric(M)

    # Circular-ish orbit at r=10M
    state = np.array([0.0, 10.0, math.pi / 2, 0.0,
                       1.0, 0.0, 0.0, 0.02], dtype=np.float64)
    state = normalize_4velocity(state, metric, -1.0)

    integrator = GeodesicIntegrator(
        metric=metric,
        particle_type=ParticleType.TIMELIKE,
        escape_radius=500.0,
        normalize_every=5,
        record_every=10,
    )

    result = integrator.integrate(state, max_affine=200.0, max_steps=30000)

    E_i = result.constants_initial['E']
    E_f = result.constants_final['E']
    L_i = result.constants_initial['L']
    L_f = result.constants_final['L']

    dE = abs(E_f - E_i)
    dL = abs(L_f - L_i)

    assert dE < 1e-4, f"Energy drift: {dE:.2e}  (E_i={E_i}, E_f={E_f})"
    assert dL < 1e-4, f"Ang. momentum drift: {dL:.2e}  (L_i={L_i}, L_f={L_f})"

    print(f"  ✓ Constants conserved: ΔE={dE:.2e}, ΔL={dL:.2e} "
          f"({result.steps_taken} steps)")


def test_photon_deflection_weak_field():
    """In the weak-field limit (b >> M), deflection → 4M/b."""
    from astrolab.physics.observables import deflection_angle_weak_field

    M = 1.0

    # At b=100M, weak-field deflection = 4/100 = 0.04 rad
    b = 100.0
    delta_phi = deflection_angle_weak_field(b, M)
    expected = 4.0 / b
    assert abs(delta_phi - expected) < 1e-12, f"δφ = {delta_phi}, expected {expected}"

    # At b=1000M
    b = 1000.0
    delta_phi = deflection_angle_weak_field(b, M)
    expected = 4.0 / b
    assert abs(delta_phi - expected) < 1e-12

    print("  ✓ Weak-field photon deflection: Δφ = 4M/b verified")


def test_effective_potential():
    """Test effective potential shape for Schwarzschild."""
    from astrolab.physics.observables import effective_potential_timelike

    M = 1.0
    L = 4.0  # Angular momentum

    # At r → ∞, Veff → 1 (for normalized potential)
    V_far = effective_potential_timelike(100.0, L, M)
    assert abs(V_far - 1.0) < 0.02, f"V_eff(r=100) = {V_far}, expected close to 1"

    # Potential should have a minimum (stable orbit) somewhere between 6M and 20M
    r_values = np.linspace(4.0, 50.0, 1000)
    V_values = [effective_potential_timelike(r, L, M) for r in r_values]

    min_idx = np.argmin(V_values)
    r_min = r_values[min_idx]
    assert 4.0 < r_min < 30.0, f"Potential minimum at r={r_min}M"

    print(f"  ✓ Effective potential: minimum at r={r_min:.2f}M, V_eff(∞)≈1")


def test_create_metric_factory():
    """Test the metric factory function."""
    from astrolab.physics.metrics import create_metric

    # Schwarzschild
    m1 = create_metric(mass_kg=1.989e30, metric_type='schwarzschild')
    assert m1.name == "Schwarzschild"
    assert m1.spin == 0.0

    # Kerr
    m2 = create_metric(mass_kg=1.989e30, metric_type='kerr', spin=0.5)
    assert m2.name == "Kerr"
    assert abs(m2.spin - 0.5) < 1e-10

    # Mass conversion check
    from astrolab.physics.metrics import mass_to_geometric, G_SI, C_SI
    M_expected = G_SI * 1.989e30 / C_SI ** 2
    assert abs(m1.M - M_expected) < 1e-6, f"M = {m1.M}, expected {M_expected}"

    print("  ✓ Metric factory: Schwarzschild and Kerr creation verified")


def test_gr_engine_integration():
    """Test the high-level GR engine."""
    from astrolab.engine.gr_engine import GRSimulationEngine, BlackHoleConfig

    config = BlackHoleConfig(mass_kg=1.989e30)
    engine = GRSimulationEngine.from_config(config)

    # Black hole info
    info = engine.black_hole_info()
    assert info['event_horizon_outer_M'] == 2.0
    assert info['photon_sphere_M'] == 3.0
    assert info['isco_prograde_M'] == 6.0

    # Quick geodesic
    result = engine.trace_geodesic(r=10, uphi=0.02, max_steps=1000, max_affine=50)
    assert len(result.trajectory) > 0

    # Time dilation
    td = engine.compute_time_dilation(r=5)
    assert 0 < td['factor'] < 1.0

    # Redshift
    rs = engine.compute_redshift(r_emit=3, r_obs=100)
    assert rs['z'] > 0

    print(f"  ✓ GR Engine: info, geodesic ({len(result.trajectory)} pts), "
          f"time dilation, redshift all working")


def test_toolkit_gr_functions():
    """Test the toolkit GR utility functions."""
    from astrolab.core.models import CelestialBody, Vector3D
    from astrolab.physics import toolkit

    sun = CelestialBody(
        name="sun", mass=1.989e30,
        position=Vector3D(0, 0, 0), velocity=Vector3D(0, 0, 0),
        radius=6.96e8, body_type="star",
    )

    # Photon sphere
    rph = toolkit.photon_sphere_radius(sun)
    rs = toolkit.schwarzschild_radius(sun)
    assert abs(rph - 1.5 * rs) < 1e-6, f"r_ph = {rph}, expected {1.5 * rs}"

    # ISCO
    r_isco = toolkit.isco_radius(sun)
    assert abs(r_isco - 3.0 * rs) < 1e-6, f"r_isco = {r_isco}, expected {3 * rs}"

    # Time dilation
    r_test = 10.0 * rs  # Far from BH
    td = toolkit.time_dilation_factor(r_test, sun)
    expected = math.sqrt(1.0 - rs / r_test)
    assert abs(td - expected) < 1e-10

    # Redshift
    rd = toolkit.gravitational_redshift_at(3 * rs, 100 * rs, sun)
    assert rd['z'] > 0

    print("  ✓ Toolkit GR functions: photon_sphere, isco, time_dilation, redshift")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all GR physics tests."""
    print("\n" + "=" * 65)
    print("  AstroLab — General Relativity Test Suite")
    print("=" * 65 + "\n")

    tests = [
        test_schwarzschild_metric_properties,
        test_schwarzschild_christoffel,
        test_horizon_and_radii,
        test_kerr_reduces_to_schwarzschild,
        test_redshift_formula,
        test_time_dilation,
        test_photon_deflection_weak_field,
        test_effective_potential,
        test_create_metric_factory,
        test_geodesic_radial_infall,
        test_constants_conservation,
        test_gr_engine_integration,
        test_toolkit_gr_functions,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  ✗ {test_fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 65}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 65}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

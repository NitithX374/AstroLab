# AstroLab CLI — User Manual

> **Version:** 2.2.0 | **Platform:** Windows / Linux / macOS | **Language:** Python 3.10+

---

## Table of Contents

1. [What is AstroLab CLI?](#1-what-is-astrolab-cli)
2. [Getting Started](#2-getting-started)
3. [Core Concepts Explained](#3-core-concepts-explained)
   - [Celestial Body](#31-celestial-body)
   - [Simulation State](#32-simulation-state)
   - [Integrators](#33-integrators)
   - [Timestep (dt)](#34-timestep-dt)
   - [N-Body Gravity](#35-n-body-gravity)
4. [Command Reference](#4-command-reference)
   - [create body](#41-create-body)
   - [edit body](#42-edit-body)
   - [delete body](#43-delete-body)
   - [show](#44-show)
   - [simulate](#45-simulate)
   - [compute](#46-compute)
   - [set](#47-set)
   - [export / import](#48-export--import)
   - [run (batch script)](#49-run-batch-script)
   - [clear](#410-clear)
5. [Unit Reference](#5-unit-reference)
6. [Astrophysics Toolkit Explained](#6-astrophysics-toolkit-explained)
7. [Walkthrough: Earth-Sun System](#7-walkthrough-earth-sun-system)
8. [Walkthrough: Three-Body System](#8-walkthrough-three-body-system)
9. [Batch Scripting](#9-batch-scripting)
10. [Tips & Best Practices](#10-tips--best-practices)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What is AstroLab CLI?

**AstroLab CLI** is an interactive command-line astrophysics simulation platform. It lets you:

- Build a system of stars, planets, moons, or asteroids
- Simulate how they move under each other's gravity over time (N-body problem)
- Compute astrophysical quantities like escape velocity, orbital period, and Lagrange points
- Save and reload your simulation
- Run automated batch simulations from script files

It is inspired by network simulation tools like **GNS3** and **Cisco CLI** — you type commands into a prompt, build your system piece by piece, and run it.

---

## 2. Getting Started

### Launch the interactive session

```powershell
cd d:\Doc\AstroLab
python main.py
```

You will see the AstroLab banner and the prompt:

```
AstroLab>
```

### Run the built-in Earth-Sun demo

```powershell
python main.py --demo
```

This runs a complete, annotated simulation of Earth orbiting the Sun for one year, then prints the results.

### Run the test suite

```powershell
python tests/test_astrolab.py
```

---

## 3. Core Concepts Explained

### 3.1 Celestial Body

A **celestial body** is any object in your simulation — a star, planet, moon, asteroid, or even a black hole. Each body has the following properties:

| Property | Meaning | Unit |
|---|---|---|
| `name` | Unique identifier for this body | — |
| `mass` | How heavy the body is. Determines the strength of its gravity. | kg |
| `position` | Where the body is in 3D space (x, y, z coordinates) | metres (m) |
| `velocity` | How fast and in what direction the body is moving (vx, vy, vz) | metres per second (m/s) |
| `radius` | The physical size of the body. Required for collision detection and escape velocity calculations. | metres (m) |
| `type` | A label: `star`, `planet`, `moon`, `asteroid`, `black_hole`, `unknown` | — |
| `color` | A display hint (used for future visualization hooks) | — |

> **Important:** All values are stored in SI units (metres, kilograms, seconds). AstroLab converts human-friendly units like `AU` and `km/s` at the command boundary.

---

### 3.2 Simulation State

The **simulation state** is the master record of everything happening in your simulation at a given moment. It contains:

| Field | Meaning |
|---|---|
| **Bodies** | The list of all celestial bodies currently in the simulation |
| **Time** | How much simulated time has passed (in seconds) since the simulation started |
| **Step** | How many individual timesteps have been executed |
| **dt** | The size of each timestep in seconds (see below) |
| **Integrator** | Which numerical method is being used to advance the simulation |

The state is saved/loaded as a JSON file, so you can pause and resume any simulation.

---

### 3.3 Integrators

An **integrator** is the mathematical algorithm that moves your simulation forward in time. Every timestep, it:

1. Calculates the gravitational force between all bodies
2. Uses that force to update each body's velocity
3. Uses the new velocity to update each body's position

There are three integrators available, each with different trade-offs:

---

#### `euler` — Euler Integrator

**What it is:** The simplest possible integration method. It takes one "guess" at the force and uses it to advance the whole step.

**Formula:**
```
new_position = position + velocity × dt
new_velocity = velocity + acceleration × dt
```

**Best for:**
- Quick demos where you only need a rough idea of the motion
- Very short simulations (a few steps)

**Avoid for:**
- Long simulations — it will slowly gain or lose energy, causing orbits to spiral in or out over time
- Precision work

> **Analogy:** Like driving with your eyes closed for 1 second, guessing you kept a straight line, opening your eyes, and correcting. Works for short stretches, but errors add up.

---

#### `rk4` — Runge-Kutta 4 (Default)

**What it is:** A much smarter integration method. Instead of one guess, it takes **four sample points** within each timestep and combines them into a weighted average. This makes it dramatically more accurate.

**Best for:**
- Most simulations — the default and best all-round choice
- Short-to-medium term simulations (hours to years of simulated time)
- High precision per step with a reasonable computation cost

**Avoid for:**
- Very long simulations (thousands of simulated years) — small energy errors accumulate slowly but they do accumulate

> **Analogy:** Instead of guessing where you'll be in one step, you take four careful measurements within that step and compute the best average path. Much more reliable.

---

#### `verlet` — Velocity Verlet (Symplectic)

**What it is:** A special "physics-aware" integrator. While not as locally accurate as RK4, it is **symplectic** — meaning it preserves the *shape* of the orbit energy mathematically. Energy does not drift even over millions of steps.

**Best for:**
- Very long simulations (thousands of years)
- Stability studies — checking if an orbit stays bound over time
- Any case where energy conservation matters more than per-step precision

**Avoid for:**
- Situations where you need high accuracy in the short term (RK4 is better)

> **Analogy:** Like a gyroscope. It doesn't have a perfect sense of direction, but it never drifts — it stays stable and balanced forever.

---

#### Quick Comparison

| | `euler` | `rk4` | `verlet` |
|---|---|---|---|
| Accuracy per step | Low (1st order) | High (4th order) | Medium (2nd order) |
| Energy drift | Fast drift | Very slow drift | No long-term drift |
| Speed | Fastest | Moderate | Fast |
| Best for | Demos | General use | Long-term stability |
| Force evaluations per step | 1 | 4 | 2 |

---

### 3.4 Timestep (dt)

The **timestep** (`dt`) is how many simulated seconds pass in each step of the simulation.

- A **smaller dt** → more accurate results, but more steps needed, so slower
- A **larger dt** → faster simulation, but less accurate; orbits may become unstable

**Recommended values:**

| Scenario | Good dt value | Meaning |
|---|---|---|
| Planetary orbits (Earth-Sun) | `3600` | 1 simulated hour per step |
| Moon orbit | `60` | 1 simulated minute per step |
| Fast-moving asteroid | `10` or less | 10 simulated seconds per step |
| Multi-star system | `86400` | 1 simulated day per step |

> **Rule of thumb:** The fastest-moving body in your system should not travel more than ~1% of the inter-body distance in a single step.

---

### 3.5 N-Body Gravity

The simulation uses **Newtonian gravity**, which means every body pulls every other body toward it. The force between any two bodies is:

```
F = G × m₁ × m₂ / r²
```

Where:
- `G` = 6.674 × 10⁻¹¹ m³ kg⁻¹ s⁻² (gravitational constant)
- `m₁`, `m₂` = masses of the two bodies in kg
- `r` = distance between them in metres
- `F` = force in Newtons, directed along the line connecting them

With **N bodies**, every pair is evaluated. A system with 5 bodies computes 10 force pairs per step. This is called the **N-body problem** — as N grows, computation grows as N².

---

## 4. Command Reference

### 4.1 `create body`

**Purpose:** Add a new celestial body to the simulation.

**Syntax:**
```
create body <name> mass=<m> [pos=(x,y,z)] [vel=(vx,vy,vz)] [radius=<r>] [type=<t>] [color=<c>]
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `name` | Yes | Unique name for this body |
| `mass` | Yes | Mass in kg (e.g. `5.97e24`) |
| `pos` | No | 3D position as `(x,y,z)`. Default: `(0,0,0)` |
| `vel` | No | 3D velocity as `(vx,vy,vz)`. Default: `(0,0,0)` |
| `radius` | No | Physical radius in metres. Needed for `compute escape_velocity` and collision detection |
| `type` | No | Body category: `star`, `planet`, `moon`, `asteroid`, `black_hole` |
| `color` | No | Display color hint |

**Examples:**
```
create body sun mass=1.989e30 pos=(0,0,0) radius=696000000 type=star
create body earth mass=5.97e24 pos=(1AU,0,0) vel=(0,29.78km/s,0) radius=6371000 type=planet
create body asteroid mass=1e15 pos=(3AU,0,0) vel=(0,17km/s,0)
```

---

### 4.2 `edit body`

**Purpose:** Change one or more properties of a body that already exists. Only the fields you supply are changed — everything else is preserved exactly as-is.

**Syntax:**
```
edit body <name> [mass=<m>] [pos=(x,y,z)] [vel=(vx,vy,vz)] [radius=<r>] [type=<t>] [color=<c>]
```

**Examples:**
```
# Forgot to set radius when creating
edit body tau radius=695700000

# Fix a wrong velocity
edit body earth vel=(0,30000,0)

# Update several fields at once
edit body sun mass=2.0e30 type=star color=yellow
```

---

### 4.3 `delete body`

**Purpose:** Remove a body from the simulation permanently.

**Syntax:**
```
delete body <name>
```

**Example:**
```
delete body moon
```

---

### 4.4 `show`

**Purpose:** Display information about the simulation state.

**Subcommands:**

```
show state              — Table of all bodies (name, type, mass, radius, position, velocity)
show body <name>        — Detailed properties of one body
show energy             — Current kinetic, potential, and total system energy
show integrators        — List available integrators and which is currently active
```

**Example:**
```
AstroLab> show state

  Simulation Time: 0.0000e+00 s (0.00 days) | Step: 0 | dt: 3600.0 s | Integrator: rk4
  ───────────────────────────────────────
  Name           Type       Mass (kg)      Radius (m)     Position (m)               Velocity (m/s)
  sun            star       1.9890e+30     6.96e+08       (0, 0, 0)                  (0, 0, 0)
  earth          planet     5.9700e+24     6.371e+06      (1.496e+11, 0, 0)          (0, 29780, 0)
```

---

### 4.5 `simulate`

**Purpose:** Run the simulation forward in time by a given number of timesteps.

**Syntax:**
```
simulate dt=<s> steps=<n> [integrator=rk4|euler|verlet] [collisions=on|off] [log_energy=<n>]
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `dt` | — | Timestep size in seconds. Required. |
| `steps` | — | How many timesteps to run. Required. |
| `integrator` | `rk4` | Which integration algorithm to use |
| `collisions` | `on` | Whether to detect and merge colliding bodies |
| `log_energy` | `0` | Log energy every N steps and show drift report. `0` = disabled |

**Calculating how long to simulate:**

```
Total simulated time = dt × steps

Examples:
  1 day   = dt=86400  steps=1
  1 year  = dt=3600   steps=8760    (3600s × 8760 = 31,536,000s ≈ 1yr)
  10 days = dt=3600   steps=240
```

**Examples:**
```
simulate dt=3600 steps=8760                         # 1 Earth year, hourly steps, RK4
simulate dt=3600 steps=8760 integrator=verlet       # Same, Velocity Verlet
simulate dt=60   steps=43200 collisions=on          # 30 days, 1-min steps, with collisions
simulate dt=3600 steps=8760 log_energy=100          # Monitor energy drift
```

---

### 4.6 `compute`

**Purpose:** Run astrophysical calculations on bodies in the simulation. These are **stateless** — they don't change the simulation.

**Subcommands:**

#### `compute escape_velocity body=<name>`
Calculates the minimum speed needed to escape the surface of a body permanently.
- Requires `radius` to be set on the body.

```
compute escape_velocity body=earth
→ Escape velocity of 'earth': 11184.1 m/s  (11.1841 km/s)
```

---

#### `compute orbital_period body=<name> primary=<primary>`
Calculates how long it takes the body to complete one full orbit around the primary, based on current separation distance.

```
compute orbital_period body=earth primary=sun
→ 3.1554e+07 s  |  365.21 days  |  0.9999 years
```

---

#### `compute grav_force body1=<name> body2=<name>`
Calculates the gravitational force between two bodies and the direction it acts.

```
compute grav_force body1=earth body2=sun
→ Force: 3.54e+22 N
```

---

#### `compute energy`
Shows the total kinetic energy (KE), potential energy (PE), and total mechanical energy of the entire system.

- **Kinetic Energy (KE):** Energy from motion. Always positive.
- **Potential Energy (PE):** Energy stored in gravitational attraction. Always negative (bound systems).
- **Total Energy:** KE + PE. If negative, the system is gravitationally bound (bodies orbit each other). If positive, bodies will escape to infinity.

```
compute energy
→ Kinetic:   +2.6513e+33 J
→ Potential: -5.3027e+33 J
→ Total:     -2.6513e+33 J
```

---

#### `compute schwarzschild body=<name>`
Calculates the **Schwarzschild radius** of a body — the radius it would need to be compressed to in order to become a black hole.

- A body whose actual radius is *smaller* than its Schwarzschild radius IS a black hole.

```
compute schwarzschild body=sun
→ Schwarzschild radius of 'sun': 2.9541e+03 m  (295.41 cm)
```

> The Sun's actual radius is ~696,000,000 m — about 235,000× larger than its Schwarzschild radius.

---

#### `compute lagrange primary=<name> secondary=<name>`
Calculates the **5 Lagrange points** of a two-body system. These are positions where a small third body can remain in a stable (or semi-stable) position relative to the two main bodies.

| Point | Location | Stability |
|---|---|---|
| **L1** | Between the two bodies | Unstable (useful for spacecraft) |
| **L2** | Beyond the secondary, away from primary | Unstable (James Webb Space Telescope!) |
| **L3** | Behind the primary, opposite the secondary | Unstable |
| **L4** | 60° ahead of the secondary in its orbit | **Stable** (Trojan asteroids live here) |
| **L5** | 60° behind the secondary in its orbit | **Stable** (Trojan asteroids live here) |

```
compute lagrange primary=sun secondary=earth
```

---

### 4.7 `set`

**Purpose:** Change simulation parameters between runs, without starting over.

**Syntax:**
```
set dt=<seconds>
set integrator=euler|rk4|verlet
```

**Examples:**
```
set dt=60              # Change timestep to 1 minute
set integrator=verlet  # Switch to Velocity Verlet for the next simulate run
```

---

### 4.8 `export` / `import`

**Purpose:** Save the current simulation state to a file, or load a previously saved state.

The file is saved in **JSON format**, which you can open and read in any text editor.

**Syntax:**
```
export state file=<filename>
import state file=<filename>
```

**Examples:**
```
export state file=solar_system.json
import state file=solar_system.json
```

> **Tip:** Export before and after a long simulation to compare the before/after states, and to resume work later.

---

### 4.9 `run` (batch script)

**Purpose:** Execute a sequence of AstroLab commands from a text file automatically.

**Syntax:**
```
run <path/to/script.astro>
```

- Lines starting with `#` are comments and are ignored.
- Empty lines are ignored.
- Each other line is treated as one AstroLab command.

**Example (from command line):**
```powershell
python main.py --script my_simulation.astro
```

---

### 4.10 `clear`

**Purpose:** Remove all bodies from the simulation and reset the time to zero. Useful for starting fresh without restarting the program.

```
clear
```

---

## 5. Unit Reference

AstroLab accepts these unit suffixes in commands. They are automatically converted to SI units when stored.

| Suffix | Meaning | Equivalent |
|---|---|---|
| *(no suffix)* | SI unit (metres, kg, m/s) | — |
| `km` | kilometres | × 1,000 m |
| `AU` | Astronomical Unit (Earth-Sun distance) | × 149,600,000,000 m |
| `km/s` | kilometres per second | × 1,000 m/s |
| `m/s` | metres per second | × 1 |
| `kg` | kilograms | × 1 |

**Examples in commands:**
```
pos=(1AU, 0, 0)          → x = 149,600,000,000 m
vel=(0, 29.78km/s, 0)    → vy = 29,780 m/s
pos=(384.4e6, 0, 0)      → x = 384,400,000 m  (Moon-Earth distance)
radius=696000km          → radius = 696,000,000 m
```

---

## 6. Astrophysics Toolkit Explained

### Escape Velocity
> *"How fast do I need to throw something to escape this planet's gravity forever?"*

Formula: `v_escape = √(2GM/R)`

Where `G` is the gravitational constant, `M` is the body's mass, and `R` is its radius.
- Earth: ~11.2 km/s
- Moon: ~2.4 km/s
- Sun: ~617 km/s

### Orbital Period
> *"How long does one orbit take?"*

Formula: `T = 2π × √(a³ / GM)`

Where `a` is the orbital radius (separation between the two bodies) and `M` is the mass of the primary.
- Earth around Sun: ~365.25 days

### Gravitational Force
> *"How hard are these two bodies pulling on each other right now?"*

Formula: `F = G × m₁ × m₂ / r²`

The force acts along the line connecting the two bodies. Newton's Third Law applies — Earth pulls on the Sun just as hard as the Sun pulls on Earth, but the Sun's enormous mass means it barely accelerates.

### System Energy
> *"Is this system bound, and how stable is it?"*

- **Kinetic Energy (KE) = ½mv²** — Always positive. More KE = moving faster.
- **Potential Energy (PE) = −GMm/r** — Always negative. More negative = deeper in gravitational well.
- **Total Energy E = KE + PE:**
  - `E < 0` → system is gravitationally **bound** (bodies orbit each other)
  - `E > 0` → bodies have enough energy to **escape** to infinity
  - `E = 0` → bodies are at the exact **escape threshold**

### Schwarzschild Radius
> *"What size would this object need to be crushed to for gravity to trap even light?"*

Formula: `Rs = 2GM/c²`

Where `c` is the speed of light. This is an introduction to General Relativity. If a body is compressed until its actual radius equals `Rs`, it becomes a black hole.

### Lagrange Points
> *"Where can I park a spacecraft so it stays in position relative to Earth and the Sun?"*

Five special equilibrium positions that exist in every two-body system. L4 and L5 are genuinely stable — the Sun-Jupiter Trojan asteroids have been sitting at L4/L5 for billions of years.

---

## 7. Walkthrough: Earth-Sun System

A complete example, step by step.

```
# Step 1: Create the Sun at the centre, at rest
create body sun mass=1.989e30 pos=(0,0,0) radius=696000000 type=star

# Step 2: Create Earth in orbit 1 AU away, with correct orbital velocity
create body earth mass=5.97e24 pos=(1AU,0,0) vel=(0,29.78km/s,0) radius=6371000 type=planet

# Step 3: Inspect the initial state
show state
show energy

# Step 4: Check orbital theory before simulating
compute escape_velocity body=earth
compute orbital_period body=earth primary=sun
compute lagrange primary=sun secondary=earth

# Step 5: Simulate 1 year (8760 hours)
simulate dt=3600 steps=8760 integrator=rk4 log_energy=100

# Step 6: Inspect results — Earth should be back near its starting point
show state
show body earth
show energy

# Step 7: Save the result
export state file=earth_sun_1yr.json
```

**What to expect:**
- Earth's final position will be very close to `(1.496e+11, ~3e8, 0)` — almost exactly one full orbit
- Energy drift with RK4 over 1 year: `0.000000%`
- The system stays gravitationally bound (total energy remains negative)

---

## 8. Walkthrough: Three-Body System

```
# Sun + Earth + Moon (Earth-Moon-Sun system)
create body sun   mass=1.989e30 pos=(0,0,0) radius=696000000 type=star
create body earth mass=5.97e24  pos=(1AU,0,0) vel=(0,29.78km/s,0) radius=6371000 type=planet
create body moon  mass=7.342e22 pos=(1.496e11,384.4e6,0) vel=(0,30802,0) radius=1737400 type=moon

show state
compute grav_force body1=earth body2=moon
compute grav_force body1=earth body2=sun

# Simulate 6 months with energy monitoring
simulate dt=3600 steps=4380 integrator=rk4 log_energy=200

show state
show energy
```

---

## 9. Batch Scripting

You can save a sequence of commands to a `.astro` file and run them automatically.

Create a file called `solar.astro`:

```bash
# solar.astro — Earth-Sun simulation batch script

create body sun   mass=1.989e30 pos=(0,0,0) radius=696000000 type=star
create body earth mass=5.97e24  pos=(1AU,0,0) vel=(0,29.78km/s,0) radius=6371000 type=planet

compute escape_velocity body=earth
compute orbital_period  body=earth primary=sun
compute schwarzschild   body=sun

simulate dt=3600 steps=8760 integrator=rk4 log_energy=100

show state
export state file=solar_result.json
```

Run it:

```powershell
python main.py --script solar.astro
```

Or from inside the REPL:

```
AstroLab> run solar.astro
```

---

## 10. Tips & Best Practices

### Choosing the right integrator

| Simulation goal | Recommended integrator |
|---|---|
| Quick exploration or demo | `euler` |
| General planetary simulations | `rk4` ✅ |
| Multi-year stability study | `verlet` |

### Choosing the right timestep (dt)

The rule is: **the fastest-moving body should not cross more than ~1% of the system scale per step**.

```
Orbital velocity of Earth  : ~29,780 m/s
System scale (Earth-Sun)   : ~1.5e11 m
1% of scale                : ~1.5e9 m

Max dt = 1.5e9 / 29780 ≈ 50,000 s ≈ 14 hours

→ dt = 3600 (1 hour) is comfortably safe for Earth-Sun.
```

### Energy as a quality indicator

Always use `log_energy=N` in simulate, then check the drift percentage:
- **< 0.01%** → Excellent. Your dt is appropriate.
- **0.01% – 1%** → Acceptable for short simulations. Consider reducing dt.
- **> 1%** → Your timestep is too large. Reduce dt or switch to `verlet`.

### Use `edit body` instead of delete + recreate

If you made a mistake in a body's properties, use `edit body` to fix just what's wrong — faster and preserves the simulation state.

### Save often

Export your state before and after each major simulation run:
```
export state file=before_sim.json
simulate dt=3600 steps=8760
export state file=after_1yr.json
```

---

## 11. Troubleshooting

### "Body 'X' not found"
The name you typed doesn't match any body in the simulation. Use `show state` to see the exact names of all bodies.

### "mass is required"
When creating a body, `mass` is the only mandatory argument. All others have safe defaults.

### Orbit is unstable / bodies flying away
- Your `dt` is too large. Try reducing it (e.g. `dt=60` instead of `dt=3600`)
- Try switching from `euler` to `rk4` or `verlet`
- Check that your initial velocity is correct for the orbit size using `compute orbital_period`

### `compute escape_velocity` gives error
The body has no radius set. Fix it with:
```
edit body <name> radius=<value>
```

### Simulation is very slow
For large steps counts, reduce the number of bodies — the computation grows as N² where N is the number of bodies.

### Unicode display issues on Windows
AstroLab handles this automatically when you run `python main.py`. If using a custom Python wrapper, ensure your terminal is set to UTF-8:
```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

---

*AstroLab CLI — built with Python 3.12, zero external dependencies.*
*Ad astra! 🚀*

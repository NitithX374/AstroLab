import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from astrolab.core.models import CelestialBody, SimulationState
from astrolab.engine.simulator import RunResult, CollisionEvent

class AstroInterpreter:
    """
    Interfaces with Generative AI to provide human-readable, physically 
    grounded explanations of simulation results and astrophysical calculations.
    """

    SYSTEM_PROMPT = (
        "You are an expert Astrophysics Professor at a world-class university. "
        "Your goal is to explain gravitational simulations and astrophysical computations. "
        "Be rigorous, insightful, and accessible. Use clear analogies where appropriate. "
        "Ensure all explanations are grounded in modern physics. "
        "Your audience consists of students using the 'AstroLab' platform. "
        "Keep your explanations concise but deep."
    )

    def __init__(
        self,
        provider: str = "anthropic",
        model:    str = "claude-haiku-4-5-20251001",
        api_key:  str = "",
    ):
        self.provider = provider.lower()
        self.model    = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def _get_client(self):
        """Lazy-load the appropriate AI client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                if not self._api_key:
                    raise ValueError(
                        "Anthropic API key is missing. Please set it using 'set ai_key=<key>' "
                        "in the CLI or export 'ANTHROPIC_API_KEY' in your environment."
                    )
                return anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "Anthropic library not installed. Run 'pip install anthropic' to enable AI features."
                )
        else:
            raise NotImplementedError(f"AI provider '{self.provider}' is not yet supported.")

    def explain_simulation(self, result: RunResult, state: SimulationState) -> str:
        """
        Explain the results of a simulation run including collisions, 
        mergers, and numerical stability.
        """
        bodies_desc = ", ".join([f"{b.name} ({b.body_type})" for b in state.bodies])
        
        prompt = (
            f"Explain the outcome of the following N-body simulation in AstroLab:\n\n"
            f"--- Context ---\n"
            f"Initial system: {bodies_desc}\n"
            f"Duration: {result.elapsed_time:.2e} seconds\n"
            f"Timesteps: {result.steps_taken}\n"
            f"Integrator: {state.integrator.upper()}\n\n"
            f"--- Events ---\n"
            f"Number of collisions: {len(result.collisions)}\n"
        )

        for i, ev in enumerate(result.collisions):
            prompt += (f"  {i+1}. At t={ev.time:.2e}s: {ev.absorbed} was inelastically "
                       f"absorbed by {ev.survivor}. Resulting mass: {ev.new_mass:.2e} kg.\n")

        if result.energy_log:
            e0 = result.energy_log[0]['total']
            ef = result.energy_log[-1]['total']
            drift = abs((ef - e0) / e0) * 100 if e0 != 0 else 0
            prompt += f"\n--- Numerical Stability ---\nEnergy conservation drift: {drift:.6f}%\n"

        return self._query(prompt)

    def explain_body(self, body: CelestialBody) -> str:
        """Explain the physical significance and properties of a specific body."""
        prompt = (
            f"Explain the physical significance and astrophysical properties of this object:\n\n"
            f"Name: {body.name}\n"
            f"Classification: {body.body_type}\n"
            f"Mass: {body.mass:.4e} kg\n"
            f"Radius: {body.radius:.4e} m\n"
            f"Current Position: {body.position}\n"
            f"Current Velocity: {body.velocity} (speed: {body.speed:.2e} m/s)\n"
        )
        return self._query(prompt)

    def explain_compute(self, command: str, result: Any, kwargs: Dict[str, Any]) -> str:
        """
        Explain the result of a scientific calculation from the astrophysics toolkit.
        """
        prompt = (
            f"The user performed a calculation in AstroLab. Please explain the result "
            f"and its physical meaning.\n\n"
            f"Computation: {command}\n"
            f"Input parameters: {kwargs if kwargs else 'None'}\n"
            f"Result: {result}\n\n"
            f"Why is this value important in astrophysics?"
        )
        return self._query(prompt)

    def ask(
        self,
        query: str,
        state: SimulationState,
        last_result: Optional[RunResult] = None,
        last_compute: Optional[tuple] = None
    ) -> str:
        """
        Versatile entry point for AI interaction. Decides which context 
        to include based on keywords and body names in the query.
        """
        query_lower = query.lower()
        context_parts = []
        
        # 1. Check for body names
        for body in state.bodies:
            if body.name.lower() in query_lower:
                context_parts.append(
                    f"--- Body Context: {body.name} ---\n"
                    f"{json.dumps(body.to_dict(), indent=2)}"
                )

        # 2. Check for explicit keywords
        if "state" in query_lower:
            bodies_summary = [b.to_dict() for b in state.bodies]
            context_parts.append(
                f"--- Overall Simulation State ---\n"
                f"Time: {state.time:.2e}s | Step: {state.step} | Timestep: {state.dt}s\n"
                f"Active Bodies: {len(state.bodies)}\n"
                f"Bodies Detail: {json.dumps(bodies_summary, indent=2)}"
            )

        if "result" in query_lower and last_result:
            context_parts.append(
                f"--- Latest Simulation Result ---\n"
                f"Duration: {last_result.elapsed_time:.2e}s | Steps: {last_result.steps_taken}\n"
                f"Collisions: {len(last_result.collisions)} recorded."
            )

        if "compute" in query_lower and last_compute:
            cmd, res, kwargs = last_compute
            context_parts.append(
                f"--- Latest Computation Result ---\n"
                f"Command: {cmd}\n"
                f"Arguments: {kwargs}\n"
                f"Numeric Result: {res}"
            )

        # Build final prompt
        if context_parts:
            context_block = "\n\n".join(context_parts)
            full_prompt = (
                f"I am working on an astrophysics simulation in AstroLab. "
                f"Here is some relevant context from my current environment:\n\n"
                f"{context_block}\n\n"
                f"User Question: {query}\n"
            )
        else:
            # Simple mode - no extra data sent to save tokens
            full_prompt = f"User Question: {query}"

        return self._query(full_prompt)

    def _query(self, prompt: str) -> str:
        """Send a prompt to the AI provider and return the response text."""
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as exc:
            return f"\n  [!] AI Interaction Failed: {exc}\n"

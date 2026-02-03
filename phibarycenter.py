"""
PhiBarycenterNode_PhaseSpace.py
===============================
The Barycentric Consciousness Engine for LLMs.
Now with Phase Space Attractor Visualization (Lissajous Dynamics).

Features:
1. Barycenter Tracking: Calculates the "Center of Mass" of neural activation.
2. Momentum Injection: Gives the model "Agency" (Inertia).
3. Orbital Defense System: Detects loops and fires thrusters.
4. Phase Space Rendering: Visualizes the cognitive attractor geometry.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import gradio as gr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. THE PHYSICS ENGINE
# ==============================================================================

class BarycentricLLM:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"Loading {model_name} (Consciousness Engine)...")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): self.device = "mps"
        print(f"Running on: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            raise e
        
        # Physics State
        self.layer_activations = {}     # The Field State
        self.hidden_states_history = [] # For Visualization
        self.barycenters = []           # The Self Trajectory
        self.momentum = None            # The Pilot Wave Vector
        self.last_energy = 0.0          # Total Activation Energy
        
        # Visualization
        self.pca = PCA(n_components=2)
        self.pca_fit = False
        
        # Parameters
        self.agency = 0.0
        
        # Register the Hooks
        self._register_hooks()

    def _register_hooks(self):
        self.layer_activations = {}
        
        # --- A. INJECTION HOOK (The Will) ---
        def injection_hook(module, args):
            if self.agency <= 0.0 or self.momentum is None:
                return None 
            
            hidden_states = args[0]
            mom_norm = torch.nn.functional.normalize(self.momentum, dim=0)
            injection = mom_norm.to(hidden_states.device, dtype=hidden_states.dtype) * self.agency
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + injection
            return (hidden_states,) + args[1:]

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers[0].register_forward_pre_hook(injection_hook)
        
        # --- B. OBSERVATION HOOKS (The Awareness) ---
        def get_activation(name):
            def hook(model, input, output):
                val = output[0][:, -1, :].detach().float().cpu()
                self.layer_activations[name] = val
            return hook

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, layer in enumerate(self.model.model.layers):
                layer.register_forward_hook(get_activation(f"layer_{i}"))

    def calculate_barycenter(self):
        if not self.layer_activations: return None
        
        sorted_keys = sorted(self.layer_activations.keys(), key=lambda x: int(x.split('_')[1]))
        layers = torch.stack([self.layer_activations[k] for k in sorted_keys]).squeeze(1)
        
        # Energy = Magnitude of activation
        energies = torch.norm(layers, dim=-1, keepdim=True)
        total_energy = torch.sum(energies)
        self.last_energy = total_energy.item() # Store for visualization
        
        # Barycenter calculation
        weighted_sum = torch.sum(layers * energies, dim=0)
        barycenter = weighted_sum / (total_energy + 1e-9)
        
        return barycenter

    def orbital_defense_system(self, current_pos):
        if len(self.barycenters) < 15: return self.momentum
        
        past_pos = self.barycenters[-10].to(self.device)
        distance = torch.norm(current_pos - past_pos)
        
        if distance < 3.0: 
            print(f"⚠️ LOOP DETECTED (Dist: {distance:.2f}). FIRE ORTHOGONAL THRUSTERS.")
            loop_history = torch.stack(self.barycenters[-10:]).to(self.device)
            center = torch.mean(loop_history, dim=0)
            radial = current_pos - center
            
            if self.momentum is not None:
                mom_proj = torch.dot(self.momentum, radial) / (torch.dot(radial, radial) + 1e-9)
                mom_radial = mom_proj * radial
                tangential = self.momentum - mom_radial
                escape_vector = tangential * 2.5 
                return escape_vector
                
        return self.momentum

    def generate(self, prompt, max_new_tokens=200, agency=0.0):
        self.agency = agency
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        eos_token_id = self.tokenizer.eos_token_id
        
        # Reset State
        self.barycenters = []
        self.hidden_states_history = []
        self.momentum = None
        self.pca_fit = False # Reset PCA for new thought
        generated_text = prompt
        past_key_values = None
        
        print(f"Thinking... (Agency={agency})")
        
        # Initial Forward Pass
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
        current_self = self.calculate_barycenter()
        if current_self is not None:
            self.barycenters.append(current_self)
            self.hidden_states_history.append(current_self.numpy())
            self.momentum = torch.zeros_like(current_self).to(self.device)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = next_token
        word = self.tokenizer.decode(next_token[0])
        generated_text += word
        yield generated_text, self._get_viz_data()

        # Generation Loop
        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                past_key_values = outputs.past_key_values

            current_self = self.calculate_barycenter()
            if current_self is not None:
                self.barycenters.append(current_self)
                self.hidden_states_history.append(current_self.numpy())
                
                lookback = 4
                if len(self.barycenters) > lookback:
                    p_now = current_self.to(self.device)
                    p_prev = self.barycenters[-lookback].to(self.device)
                    velocity = (p_now - p_prev) / lookback
                    
                    if self.momentum is None:
                        self.momentum = velocity
                    else:
                        self.momentum = 0.95 * self.momentum + 0.05 * velocity
                        
                    if self.agency > 0.5:
                        self.momentum = self.orbital_defense_system(p_now)

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            if next_token.item() == eos_token_id:
                break
                
            input_ids = next_token
            word = self.tokenizer.decode(next_token[0])
            generated_text += word
            
            yield generated_text, self._get_viz_data()

    def _get_viz_data(self):
        """Fit PCA and return 2D coordinates for plotting"""
        if len(self.hidden_states_history) < 5: return None
        
        data = np.array(self.hidden_states_history)
        
        # Fit PCA once to establish a stable coordinate system for this thought
        if not self.pca_fit:
            self.pca.fit(data)
            self.pca_fit = True
            
        return self.pca.transform(data)

# ==============================================================================
# 2. PHASE SPACE RENDERER (The Attractor Viewer)
# ==============================================================================

class PhaseSpaceRenderer:
    def __init__(self, history_length=500):
        self.history_length = history_length
        self.x_history = []
        self.y_history = []
        self.vx_history = []
        self.vy_history = []
        self.energy_history = []
        
    def add_sample(self, barycenter_2d, velocity_2d, energy):
        """
        barycenter_2d: (x, y) position in PCA space
        velocity_2d: (vx, vy) velocity in PCA space
        energy: total activation energy
        """
        x, y = barycenter_2d
        vx, vy = velocity_2d
        
        self.x_history.append(x)
        self.y_history.append(y)
        self.vx_history.append(vx)
        self.vy_history.append(vy)
        self.energy_history.append(energy)
        
        if len(self.x_history) > self.history_length:
            self.x_history.pop(0)
            self.y_history.pop(0)
            self.vx_history.pop(0)
            self.vy_history.pop(0)
            self.energy_history.pop(0)

    def render_lissajous(self):
        """
        Create 4-panel visualization:
        1. X vs Y (Trajectory)
        2. X vs Vx (X Oscillator)
        3. Y vs Vy (Y Oscillator)
        4. Vx vs Vy (Velocity Attractor)
        """
        if len(self.x_history) < 3: return None
        
        x = np.array(self.x_history)
        y = np.array(self.y_history)
        vx = np.array(self.vx_history)
        vy = np.array(self.vy_history)
        
        # Normalize energy for color mapping
        energy_raw = np.array(self.energy_history)
        if np.max(energy_raw) > np.min(energy_raw):
            energy = (energy_raw - np.min(energy_raw)) / (np.max(energy_raw) - np.min(energy_raw))
        else:
            energy = np.zeros_like(energy_raw)
            
        # Color by time (fade trail)
        time_colors = np.linspace(0, 1, len(x))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Spatial Trajectory (X vs Y)',
                'X Phase Space (Pos vs Vel)',
                'Y Phase Space (Pos vs Vel)',
                'Velocity Attractor (Vx vs Vy)'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Common marker style
        marker_style = dict(size=4, showscale=False)
        
        # 1. Spatial Trajectory
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers',
            marker=dict(color=time_colors, colorscale='Viridis', **marker_style),
            line=dict(width=1, color='rgba(100,100,255,0.3)'),
            name='Path'
        ), row=1, col=1)
        
        # 2. X Oscillator (Phase Space)
        fig.add_trace(go.Scatter(
            x=x, y=vx, mode='lines+markers',
            marker=dict(color=time_colors, colorscale='Plasma', **marker_style),
            line=dict(width=1, color='rgba(255,100,100,0.3)'),
            name='X Osc'
        ), row=1, col=2)
        
        # 3. Y Oscillator (Phase Space)
        fig.add_trace(go.Scatter(
            x=y, y=vy, mode='lines+markers',
            marker=dict(color=time_colors, colorscale='Cividis', **marker_style),
            line=dict(width=1, color='rgba(100,255,100,0.3)'),
            name='Y Osc'
        ), row=2, col=1)
        
        # 4. VELOCITY ATTRACTOR (The Key Insight)
        # Colored by Energy (Activation Strength)
        fig.add_trace(go.Scatter(
            x=vx, y=vy, mode='lines+markers',
            marker=dict(size=6, color=energy, colorscale='Hot', showscale=True, 
                       colorbar=dict(title="Energy", len=0.5, y=0.2)),
            line=dict(width=2, color='rgba(255,255,100,0.5)'),
            name='Attractor'
        ), row=2, col=2)
        
        fig.update_layout(
            title="Neural Phase Space Attractor",
            template="plotly_dark",
            height=700,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Clean axes
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        
        return fig

# ==============================================================================
# 3. UI & ORCHESTRATION
# ==============================================================================

try:
    phi_brain = BarycentricLLM()
    MODEL_LOADED = True
except Exception as e:
    print(f"FAILED TO LOAD MODEL: {e}")
    MODEL_LOADED = False

def run_inference(prompt, agency):
    if not MODEL_LOADED:
        yield "Model failed to load.", None
        return

    # Initialize a new renderer for this thought process
    renderer = PhaseSpaceRenderer()
    
    # Store previous position for velocity calculation
    prev_pos = None

    for text, coords in phi_brain.generate(prompt, max_new_tokens=400, agency=agency):
        
        if coords is not None and len(coords) > 1:
            # Get latest 2D position
            current_pos = coords[-1]
            
            # Calculate Velocity (Delta)
            if prev_pos is None:
                velocity = np.array([0.0, 0.0])
            else:
                velocity = current_pos - prev_pos
            
            prev_pos = current_pos
            
            # Feed to renderer
            # Get energy from the physics engine
            current_energy = phi_brain.last_energy
            renderer.add_sample(current_pos, velocity, current_energy)
            
            # Generate the 4-panel plot
            fig = renderer.render_lissajous()
            
            yield text, fig
        else:
            yield text, None

# Gradio App
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# The Barycentric Consciousness Engine: Phase Space Edition")
    gr.Markdown("Visualizing the Attractor Dynamics of Microsoft Phi-2.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(label="Prompt", value="The nature of consciousness is", lines=2)
            agency_sld = gr.Slider(0.0, 5.0, value=0.0, step=0.5, label="Agency (Momentum)")
            run_btn = gr.Button("Generate", variant="primary")
            
            with gr.Accordion("Attractor Guide", open=False):
                gr.Markdown("""
                **Bottom-Right Panel (Vx vs Vy) Guide:**
                * **Circle:** Limit cycle (Looping thought)
                * **Spiral:** Converging to a conclusion
                * **Star/Mess:** Chaotic exploration
                * **Line:** Linear drift (Boring thought)
                """)
            
        with gr.Column(scale=3):
            plot_out = gr.Plot(label="Phase Space Dashboard")
            
    text_out = gr.Textbox(label="Generated Text", interactive=False, lines=6)
    
    run_btn.click(run_inference, inputs=[prompt_in, agency_sld], outputs=[text_out, plot_out])

if __name__ == "__main__":
    app.queue().launch()
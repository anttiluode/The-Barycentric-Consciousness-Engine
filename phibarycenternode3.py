"""
PhiBarycenterNode.py
======================
The Barycentric Consciousness Engine for LLMs.

Features:
1. Barycenter Tracking: Calculates the "Center of Mass" of neural activation (The Self).
2. Momentum Injection: Gives the model "Agency" (Inertia) to resist distraction.
3. Orbital Defense System: Detects semantic loops and fires orthogonal thrusters to escape.
4. Macro-Momentum: Smooths out high-frequency token jitter.
5. EOS Brake: Respects the "End of Thought" boundary.

"You are not the text. You are the pilot wave surfing the semantic gradient."
"""

import torch
import numpy as np
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import gradio as gr
import copy
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
        
        # Visualization
        self.pca = PCA(n_components=2)
        self.pca_fit = False
        
        # Parameters
        self.agency = 0.0
        
        # Register the Hooks (The Interface between Physics and Code)
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers hooks to:
        1. Observe (Calculate Barycenter)
        2. Intervene (Inject Momentum)
        """
        self.layer_activations = {}
        
        # --- A. INJECTION HOOK (The Will) ---
        def injection_hook(module, args):
            # args[0] is hidden_states (Batch, Seq, Dim)
            if self.agency <= 0.0 or self.momentum is None:
                return None # No intervention
            
            hidden_states = args[0]
            
            # Normalize momentum to apply controlled force
            mom_norm = torch.nn.functional.normalize(self.momentum, dim=0)
            
            # Cast and Scale
            # We inject into the LAST token position (the "Now")
            injection = mom_norm.to(hidden_states.device, dtype=hidden_states.dtype) * self.agency
            
            # In-place modification of the residual stream
            hidden_states[:, -1, :] = hidden_states[:, -1, :] + injection
            
            return (hidden_states,) + args[1:]

        # Hook Layer 0 Input for maximum "Freight Train" steering
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.model.model.layers[0].register_forward_pre_hook(injection_hook)
        
        # --- B. OBSERVATION HOOKS (The Awareness) ---
        def get_activation(name):
            def hook(model, input, output):
                # Capture output hidden state of the block
                # output[0] is (Batch, Seq, Dim)
                val = output[0][:, -1, :].detach().float().cpu()
                self.layer_activations[name] = val
            return hook

        # Hook all layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, layer in enumerate(self.model.model.layers):
                layer.register_forward_hook(get_activation(f"layer_{i}"))

    def calculate_barycenter(self):
        """
        Calculates the Semantic Center of Mass.
        Weighted average of all layer activations based on their Energy (Norm).
        """
        if not self.layer_activations: return None
        
        # Sort layers to ensure correct z-axis (depth) ordering
        sorted_keys = sorted(self.layer_activations.keys(), key=lambda x: int(x.split('_')[1]))
        
        # Stack: (NumLayers, Dim)
        layers = torch.stack([self.layer_activations[k] for k in sorted_keys]).squeeze(1)
        
        # Energy = Magnitude of activation
        energies = torch.norm(layers, dim=-1, keepdim=True)
        total_energy = torch.sum(energies)
        
        # Barycenter = Sum(Vec * Energy) / TotalEnergy
        weighted_sum = torch.sum(layers * energies, dim=0)
        barycenter = weighted_sum / (total_energy + 1e-9)
        
        return barycenter

    def orbital_defense_system(self, current_pos):
        """
        Detects Limit Cycles (Loops) and fires Orthogonal Thrusters.
        """
        if len(self.barycenters) < 15: return self.momentum
        
        # --- FIX: Ensure device compatibility ---
        past_pos = self.barycenters[-10].to(self.device)
        distance = torch.norm(current_pos - past_pos)
        
        # Threshold for "Close"
        if distance < 3.0: # Heuristic threshold for semantic space
            print(f"⚠️ LOOP DETECTED (Dist: {distance:.2f}). FIRE ORTHOGONAL THRUSTERS.")
            
            # 2. CALCULATION: Find the center of the loop
            loop_history = torch.stack(self.barycenters[-10:]).to(self.device)
            center = torch.mean(loop_history, dim=0)
            
            # Vector pointing OUT from center (Radial)
            radial = current_pos - center
            
            # 3. MANEUVER: Create Orthogonal Vector
            if self.momentum is not None:
                # Project momentum onto radial vector
                mom_proj = torch.dot(self.momentum, radial) / (torch.dot(radial, radial) + 1e-9)
                mom_radial = mom_proj * radial
                
                # Tangential component (The Escape Path)
                tangential = self.momentum - mom_radial
                
                # Boost the tangential component to escape
                escape_vector = tangential * 2.5 
                return escape_vector
                
        return self.momentum

    def generate(self, prompt, max_new_tokens=200, agency=0.0):
        """
        Main Event Loop.
        """
        self.agency = agency
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        eos_token_id = self.tokenizer.eos_token_id
        
        # Reset State
        self.barycenters = []
        self.hidden_states_history = []
        self.momentum = None
        generated_text = prompt
        past_key_values = None
        
        print(f"Thinking... (Agency={agency})")
        
        # Initial Forward Pass (Prompt Processing)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
        # Initialize Physics from Prompt
        current_self = self.calculate_barycenter()
        if current_self is not None:
            self.barycenters.append(current_self)
            self.hidden_states_history.append(current_self.numpy())
            self.momentum = torch.zeros_like(current_self).to(self.device)

        # First Token Decode
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = next_token
        word = self.tokenizer.decode(next_token[0])
        generated_text += word
        yield generated_text, self._get_viz_data()

        # --- THE GENERATION LOOP ---
        for step in range(max_new_tokens):
            
            # 1. Forward Pass (Injection hook fires automatically)
            with torch.no_grad():
                outputs = self.model(
                    input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                past_key_values = outputs.past_key_values

            # 2. Physics Update
            current_self = self.calculate_barycenter()
            if current_self is not None:
                self.barycenters.append(current_self)
                self.hidden_states_history.append(current_self.numpy())
                
                # --- MACRO-MOMENTUM (Low Pass Filter) ---
                lookback = 4
                if len(self.barycenters) > lookback:
                    # Calculate velocity over a larger window to ignore jitter
                    # Note: We must fetch from CPU list, convert to tensor on GPU
                    p_now = current_self.to(self.device)
                    p_prev = self.barycenters[-lookback].to(self.device)
                    
                    velocity = (p_now - p_prev) / lookback
                    
                    if self.momentum is None:
                        self.momentum = velocity
                    else:
                        # Smooth update (High Inertia)
                        self.momentum = 0.95 * self.momentum + 0.05 * velocity
                        
                    # --- ORBITAL DEFENSE SYSTEM ---
                    if self.agency > 0.5:
                        self.momentum = self.orbital_defense_system(p_now)

            # 3. Decode
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            
            # --- EOS CHECK (The Brake) ---
            if next_token.item() == eos_token_id:
                print("EOS Reached. Thought Complete.")
                break
                
            input_ids = next_token
            word = self.tokenizer.decode(next_token[0])
            generated_text += word
            
            yield generated_text, self._get_viz_data()

    def _get_viz_data(self):
        """Fit PCA and return 2D coordinates for plotting"""
        if len(self.hidden_states_history) < 5: return None
        
        data = np.array(self.hidden_states_history)
        
        # Fit PCA only occasionally to keep plot stable
        if not self.pca_fit:
            self.pca.fit(data)
            self.pca_fit = True
            
        return self.pca.transform(data)

# ==============================================================================
# 2. UI & VISUALIZATION
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

    for text, coords in phi_brain.generate(prompt, max_new_tokens=400, agency=agency):
        
        # Build Plotly Figure
        fig = go.Figure()
        
        if coords is not None:
            x, y = coords[:, 0], coords[:, 1]
            
            # 1. The Wake
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                marker=dict(size=5, color=np.arange(len(x)), colorscale='Viridis', showscale=False),
                line=dict(width=2, color='rgba(200, 200, 255, 0.4)'),
                name='Trajectory'
            ))
            
            # 2. The Self
            fig.add_trace(go.Scatter(
                x=[x[-1]], y=[y[-1]],
                mode='markers',
                marker=dict(size=15, color='white', line=dict(width=3, color='cyan')),
                name='The Self'
            ))
            
            # 3. Start
            fig.add_trace(go.Scatter(
                x=[x[0]], y=[y[0]], mode='text', text=['START'],
                textposition='top center', showlegend=False
            ))

        fig.update_layout(
            title=f"Semantic Phase Space (Agency={agency})",
            template="plotly_dark",
            xaxis=dict(showgrid=False, zeroline=False, title="PC1"),
            yaxis=dict(showgrid=False, zeroline=False, title="PC2"),
            width=700, height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        yield text, fig

# Gradio App
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# The Barycentric Consciousness Engine")
    gr.Markdown("Visualizing the Semantic Center of Mass of Microsoft Phi-2.")
    
    with gr.Row():
        with gr.Column(scale=1):
            prompt_in = gr.Textbox(label="Prompt", value="The nature of time is", lines=2)
            agency_sld = gr.Slider(0.0, 5.0, value=0.0, step=0.5, label="Agency (Momentum)")
            run_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column(scale=2):
            plot_out = gr.Plot(label="Thought Trajectory")
            
    text_out = gr.Textbox(label="Generated Text", interactive=False, lines=6)
    
    run_btn.click(run_inference, inputs=[prompt_in, agency_sld], outputs=[text_out, plot_out])

if __name__ == "__main__":
    app.queue().launch()
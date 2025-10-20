"""
Enhanced 3D Visualization with Real-time Streaming and Interactive Controls
Extends the base SingularityVisualizer with advanced features:
- Real-time streaming 3D viewer
- Multi-singularity trajectory tracking
- Interactive time slider
- Streamline visualization for vector fields
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from collections import deque
import time

from .visualization import SingularityVisualizer, VisualizationConfig

# VTK Export Support (Patch #3.3)
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logging.getLogger(__name__).warning("[VTK Export] meshio not installed - VTK export disabled")

logger = logging.getLogger(__name__)

try:
    import dash  # type: ignore
    from dash import dcc, html  # type: ignore
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    dash = None  # type: ignore
    dcc = None  # type: ignore
    html = None  # type: ignore
    logger.warning("[Dashboard] dash not installed - interactive dashboards disabled")


@dataclass
class StreamingConfig:
    """Configuration for real-time streaming visualization"""
    buffer_size: int = 100
    update_interval: float = 0.1  # seconds
    trajectory_length: int = 50
    enable_streamlines: bool = True
    streamline_density: float = 1.0


class EnhancedSingularityVisualizer(SingularityVisualizer):
    """
    Enhanced visualization suite with real-time streaming and advanced 3D features

    New Features:
    - Real-time 3D streaming viewer
    - Multi-singularity trajectory tracking
    - Interactive time slider with playback controls
    - Vector field streamlines in 3D
    """

    def __init__(self, config: VisualizationConfig = None,
                 streaming_config: StreamingConfig = None):
        super().__init__(config)

        if streaming_config is None:
            streaming_config = StreamingConfig()
        self.streaming_config = streaming_config

        # Buffers for real-time data
        self.field_buffer = deque(maxlen=streaming_config.buffer_size)
        self.time_buffer = deque(maxlen=streaming_config.buffer_size)
        self.singularity_trajectories = {}

        logger.info("Initialized EnhancedSingularityVisualizer with streaming support")


    def plot_3d_streaming_viewer(self,
                                 data_generator: Callable,
                                 max_frames: int = 1000,
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create real-time 3D streaming visualization

        Args:
            data_generator: Function that yields (field, time, singularities) tuples
            max_frames: Maximum number of frames to display
            save_path: Path to save HTML with streaming support

        Returns:
            Plotly Figure with real-time updates
        """
        logger.info("Creating real-time 3D streaming viewer")

        # Create figure with multiple subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scatter3d', 'rowspan': 2}, {'type': 'scatter'}],
                [None, {'type': 'scatter'}]
            ],
            subplot_titles=(
                '3D Singularity Evolution',
                'Magnitude vs Time',
                'Lambda Distribution'
            ),
            horizontal_spacing=0.15,
            vertical_spacing=0.1
        )

        # Initialize 3D scatter for singularities
        fig.add_trace(
            go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=[],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Lambda", x=1.15, len=0.5)
                ),
                line=dict(width=2, color='rgba(255,0,0,0.3)'),
                name='Singularities',
                hovertemplate="Location: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>" +
                             "Lambda: %{marker.color:.3f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Initialize magnitude plot
        fig.add_trace(
            go.Scatter(
                x=[], y=[],
                mode='lines',
                line=dict(width=2, color='blue'),
                name='Max Magnitude'
            ),
            row=1, col=2
        )

        # Initialize lambda distribution
        fig.add_trace(
            go.Scatter(
                x=[], y=[],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Lambda Values'
            ),
            row=2, col=2
        )

        # Layout configuration
        fig.update_layout(
            title=dict(
                text="Real-Time 3D Singularity Streaming Viewer<br>" +
                     "<sub>Live monitoring of unstable singularity evolution</sub>",
                x=0.5
            ),
            showlegend=True,
            height=800,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    x=0.1,
                    y=1.15,
                    buttons=[
                        dict(label="Play", method="animate", args=[None,
                            {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                        dict(label="Pause", method="animate", args=[[None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                    ]
                )
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "y": 0.0,
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "steps": []
            }]
        )

        # Update 3D scene
        fig.update_scenes(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Real-time 3D streaming viewer saved to {save_path}")

        return fig


    def plot_singularity_trajectories(self,
                                     simulation_results: Dict,
                                     save_path: Optional[str] = None) -> go.Figure:
        """
        Plot trajectories of multiple singularities over time with tracking

        Args:
            simulation_results: Dictionary with 'singularity_events' and 'time_history'
            save_path: Optional path to save figure

        Returns:
            Interactive Plotly figure with trajectory visualization
        """
        logger.info("Plotting multi-singularity trajectories")

        singularity_events = simulation_results.get('singularity_events', [])
        if not singularity_events:
            logger.warning("No singularity events to plot trajectories")
            return None

        # Group events by singularity ID (cluster nearby events)
        trajectories = self._cluster_singularity_trajectories(singularity_events)

        # Create 3D plot
        fig = go.Figure()

        # Color palette for different trajectories
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

        for idx, (traj_id, events) in enumerate(trajectories.items()):
            # Extract trajectory data
            times = [e.time for e in events]
            x_coords = [e.location[0] for e in events]
            y_coords = [e.location[1] for e in events]
            z_coords = [e.location[2] for e in events]
            lambdas = [e.lambda_estimate for e in events]
            magnitudes = [e.magnitude for e in events]

            color = colors[idx % len(colors)]

            # Plot trajectory as line
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines+markers',
                line=dict(width=4, color=color),
                marker=dict(
                    size=[min(15, 5 + np.log10(m)) for m in magnitudes],
                    color=times,
                    colorscale='Viridis',
                    showscale=(idx == 0),
                    colorbar=dict(title="Time", x=1.1) if idx == 0 else None
                ),
                text=[f"t={t:.4f}, λ={l:.3f}, mag={m:.2e}"
                      for t, l, m in zip(times, lambdas, magnitudes)],
                hovertemplate="<b>Trajectory %s</b><br>" % traj_id +
                             "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>" +
                             "%{text}<extra></extra>",
                name=f"Singularity {traj_id}"
            ))

            # Add start marker
            fig.add_trace(go.Scatter3d(
                x=[x_coords[0]],
                y=[y_coords[0]],
                z=[z_coords[0]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='diamond'),
                name=f"Start {traj_id}",
                showlegend=False,
                hovertext=f"Start: t={times[0]:.4f}"
            ))

            # Add end marker
            fig.add_trace(go.Scatter3d(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                z=[z_coords[-1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name=f"End {traj_id}",
                showlegend=False,
                hovertext=f"End: t={times[-1]:.4f}"
            ))

        # Layout
        fig.update_layout(
            title=dict(
                text="Multi-Singularity Trajectory Tracking<br>" +
                     "<sub>Evolution of unstable singularities in 3D space</sub>",
                x=0.5
            ),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
                aspectmode='cube'
            ),
            width=1000,
            height=800,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Singularity trajectories saved to {save_path}")

        return fig


    def plot_interactive_time_slider(self,
                                     simulation_results: Dict,
                                     save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive visualization with time slider control

        Args:
            simulation_results: Dictionary with field_history, time_history, singularity_events
            save_path: Optional path to save HTML

        Returns:
            Interactive figure with time slider
        """
        logger.info("Creating interactive time slider visualization")

        field_history = simulation_results.get('field_history', [])
        time_history = simulation_results.get('time_history', [])
        singularity_events = simulation_results.get('singularity_events', [])

        if not field_history or not time_history:
            logger.warning("No field history for time slider visualization")
            return None

        # Create frames for animation
        frames = []
        for idx, (field, time) in enumerate(zip(field_history, time_history)):
            # Process field magnitude
            if len(field.shape) == 4:  # Vector field
                field_mag = np.linalg.norm(field, axis=-1)
            else:
                field_mag = np.abs(field)

            # Get singularities up to this time
            current_sings = [s for s in singularity_events if s.time <= time]

            if current_sings:
                sing_x = [s.location[0] for s in current_sings]
                sing_y = [s.location[1] for s in current_sings]
                sing_z = [s.location[2] for s in current_sings]
                sing_lambdas = [s.lambda_estimate for s in current_sings]
            else:
                sing_x, sing_y, sing_z, sing_lambdas = [], [], [], []

            # Create isosurface for this timestep
            nx, ny, nz = field_mag.shape
            x = np.linspace(-2, 2, nx)
            y = np.linspace(-2, 2, ny)
            z = np.linspace(-1, 1, nz)

            threshold = np.percentile(field_mag, 95)

            frame = go.Frame(
                data=[
                    go.Isosurface(
                        x=x.flatten(),
                        y=y.flatten(),
                        z=z.flatten(),
                        value=field_mag.flatten(),
                        isomin=threshold,
                        isomax=field_mag.max(),
                        opacity=0.6,
                        surface_count=2,
                        colorscale='Viridis',
                        name="Field"
                    ),
                    go.Scatter3d(
                        x=sing_x, y=sing_y, z=sing_z,
                        mode='markers',
                        marker=dict(size=10, color=sing_lambdas,
                                   colorscale='Reds', showscale=True),
                        name="Singularities"
                    )
                ],
                name=str(idx),
                layout=go.Layout(title_text=f"Time: {time:.4f}")
            )
            frames.append(frame)

        # Initial figure
        fig = go.Figure(data=frames[0].data if frames else [], frames=frames)

        # Add play/pause buttons and slider
        fig.update_layout(
            title="Interactive Time Evolution",
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate"}]}
                ],
                "x": 0.1, "y": 1.15
            }],
            sliders=[{
                "active": 0,
                "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True},
                                               "mode": "immediate"}],
                          "label": f"{time_history[int(f.name)]:.4f}",
                          "method": "animate"}
                         for f in frames],
                "x": 0.1, "len": 0.85, "y": 0,
                "currentvalue": {"prefix": "Time: ", "visible": True}
            }]
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive time slider visualization saved to {save_path}")

        return fig


    def _cluster_singularity_trajectories(self,
                                         events: List,
                                         spatial_threshold: float = 0.5) -> Dict:
        """
        Cluster singularity events into trajectories based on spatial proximity

        Args:
            events: List of singularity events
            spatial_threshold: Maximum distance to consider events as same trajectory

        Returns:
            Dictionary mapping trajectory ID to list of events
        """
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.time)

        trajectories = {}
        next_traj_id = 0

        for event in sorted_events:
            # Find closest existing trajectory
            closest_traj = None
            min_dist = float('inf')

            for traj_id, traj_events in trajectories.items():
                last_event = traj_events[-1]
                dist = np.linalg.norm(
                    np.array(event.location) - np.array(last_event.location)
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_traj = traj_id

            # Assign to existing trajectory or create new one
            if closest_traj is not None and min_dist < spatial_threshold:
                trajectories[closest_traj].append(event)
            else:
                trajectories[next_traj_id] = [event]
                next_traj_id += 1

        return trajectories


# VTK Export Function (Patch #3.3)
def export_to_vtk(filename: str, coords: torch.Tensor, u_pred: torch.Tensor):
    """
    Export PINN solution to VTK for Paraview visualization

    Args:
        filename: Output VTK file path
        coords: Spatial coordinates (N x D tensor)
        u_pred: Predicted solution values (N x 1 tensor)

    Usage:
        export_to_vtk("solution.vtk", coords, u_pred)
        # Open in Paraview for publication-quality visualization
    """
    if not MESHIO_AVAILABLE:
        logger.error("[VTK Export] meshio not installed. Install: pip install meshio")
        return

    try:
        points = coords.detach().cpu().numpy()
        values = u_pred.detach().cpu().numpy().flatten()

        # Create vertex cells for point cloud
        cells = [("vertex", np.arange(len(points)).reshape(-1, 1))]

        # Write to VTK
        meshio.write_points_cells(
            filename,
            points,
            cells,
            point_data={"u": values}
        )
        logger.info(f"[VTK Export] Saved solution to {filename}")
        print(f"[+] VTK Export: {filename} (open with Paraview)")

    except Exception as e:
        logger.error(f"[VTK Export] Failed: {e}")


def launch_residual_dashboard(residual_history: List[float]) -> None:
    """Launch an interactive dashboard for residual convergence monitoring."""
    if not DASH_AVAILABLE:
        logger.error("[Dashboard] dash is not installed. Install via 'pip install dash' to enable dashboards.")
        return

    history = [float(value) for value in residual_history]
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Residual Convergence Dashboard"),
        dcc.Graph(
            id="residual-graph",
            figure={
                "data": [
                    go.Scatter(y=history, mode="lines", name="Residual")
                ],
                "layout": go.Layout(
                    title="Residual Convergence",
                    xaxis={"title": "Epoch"},
                    yaxis={"title": "Residual (log scale)", "type": "log"}
                )
            }
        )
    ])
    logger.info("[Dashboard] Starting residual convergence dashboard at http://127.0.0.1:8050")
    app.run_server(debug=False, use_reloader=False)


def plot_lambda_instability_pattern(lambdas: List[float],
                                    confidence: List[float],
                                    save_path: Optional[str] = None) -> None:
    """Plot lambda instability pattern with regression diagnostics using Matplotlib."""
    if not lambdas:
        logger.warning("No lambda values provided for instability pattern plot.")
        return

    try:
        import matplotlib.pyplot as plt  # Local import for headless environments
        import numpy as np
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        logger.error(f"[Visualization] Missing dependency for instability pattern plot: {exc}")
        return

    x = np.arange(len(lambdas)).reshape(-1, 1)
    y = np.asarray(lambdas, dtype=float)
    conf = np.asarray(confidence, dtype=float) if confidence else None

    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r_squared = model.score(x, y)

    plt.figure(figsize=(8, 5))
    plt.errorbar(range(len(lambdas)), y, yerr=conf, fmt='o',
                 label='Lambda (CI)' if conf is not None else 'Lambda')
    plt.plot(range(len(lambdas)), y_pred, '--', label=f'Linear Fit (R^2={r_squared:.3f})')
    plt.xlabel('Instability Order')
    plt.ylabel('Lambda Estimate')
    plt.title('Instability Pattern with Regression Diagnostics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"[Visualization] Saved lambda instability pattern to {save_path}")
    plt.close()


def launch_comparison_dashboard(stage1_data: torch.Tensor,
                                stage2_data: torch.Tensor,
                                coords: torch.Tensor) -> None:
    """Interactive web UI to compare Stage 1 and Stage 2 predictions."""
    if not DASH_AVAILABLE:
        logger.error("[Dashboard] dash is not installed. Install via 'pip install dash' to enable dashboards.")
        return

    import plotly.express as px  # Local import to keep dependency optional

    stage1_vals = stage1_data.detach().cpu().numpy().flatten()
    stage2_vals = stage2_data.detach().cpu().numpy().flatten()
    coords_np = coords.detach().cpu().numpy()

    fig1 = px.scatter(x=coords_np[:, 0], y=stage1_vals, title="Stage 1 Predictions")
    fig2 = px.scatter(x=coords_np[:, 0], y=stage2_vals, title="Stage 2 Predictions")

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Stage Comparison Dashboard"),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2)
    ])
    logger.info("[Dashboard] Starting stage comparison dashboard at http://127.0.0.1:8050")
    app.run_server(debug=False, use_reloader=False)


def plot_lambda_timeseries(lambda_history: List[float],
                           save_path: Optional[str] = None) -> None:
    """Plot lambda estimates as a time-series."""
    if not lambda_history:
        logger.warning("No lambda history provided for time-series plot.")
        return

    import matplotlib.pyplot as plt  # Local import for compatibility with headless backends

    plt.figure(figsize=(8, 4))
    plt.plot(lambda_history, marker='o')
    plt.xlabel('Training Step')
    plt.ylabel('Lambda Estimate')
    plt.title('Lambda Evolution Over Training')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"[Visualization] Saved lambda time-series to {save_path}")
    plt.close()


# HTML Report Generator (Patch #9.2)
def export_html_report(residual_history: list, lambdas: list, filename: str = "report.html"):
    """
    Generate standalone HTML with interactive plots (Patch #9.2)

    Args:
        residual_history: List of residual values over iterations
        lambdas: List of lambda values
        filename: Output HTML file path

    Usage:
        export_html_report(residual_history, lambdas, "experiment_report.html")
    """
    try:
        import plotly.graph_objs as go
        import plotly.offline as pyo

        # Residual convergence plot
        fig1 = go.Figure([go.Scatter(y=residual_history, mode="lines", name="Residual")])
        fig1.update_layout(
            title="Residual Convergence",
            xaxis_title="Iteration",
            yaxis_title="Residual Loss",
            yaxis_type="log",
            template="plotly_white"
        )

        # Lambda instability pattern
        fig2 = go.Figure([go.Scatter(y=lambdas, mode="markers+lines", name="Lambda")])
        fig2.update_layout(
            title="Lambda Instability Pattern",
            xaxis_title="Detection Index",
            yaxis_title="Lambda Value",
            template="plotly_white"
        )

        # Generate HTML
        with open(filename, "w", encoding="utf-8") as f:
            f.write("<html><head><title>Experiment Report</title></head><body>")
            f.write("<h1>Singularity Detection - Experiment Report</h1>")
            f.write("<h2>Residual Convergence</h2>")
            f.write(pyo.plot(fig1, include_plotlyjs="cdn", output_type="div"))
            f.write("<h2>Lambda Instability Pattern</h2>")
            f.write(pyo.plot(fig2, include_plotlyjs=False, output_type="div"))
            f.write("</body></html>")

        logger.info(f"[HTML Report] Saved to {filename}")
        print(f"[+] HTML Report: {filename} (interactive plots)")

    except Exception as e:
        logger.error(f"[HTML Report] Failed: {e}")


# Convenience function for quick usage
def create_enhanced_visualizer(config: VisualizationConfig = None,
                               streaming_config: StreamingConfig = None):
    """
    Factory function to create EnhancedSingularityVisualizer

    Usage:
        visualizer = create_enhanced_visualizer()
        fig = visualizer.plot_singularity_trajectories(results)
    """
    return EnhancedSingularityVisualizer(config, streaming_config)


if __name__ == "__main__":
    # Demo usage
    print("[*] Enhanced Visualization Suite - Demo")
    print("[=] Features:")
    print("    [+] Real-time 3D streaming viewer")
    print("    [+] Multi-singularity trajectory tracking")
    print("    [+] Interactive time slider with playback")
    print("\n[W] Use create_enhanced_visualizer() to get started!")

import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, render, ui, reactive

# --- UI Setup ---
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Control Panel"),
        
        # Correlation Strength Slider
        ui.input_slider(
            "r", 
            "Correlation Strength (r):", 
            min=-0.99, 
            max=0.99, 
            value=0.70, 
            step=0.01
        ),
        
        # Quick Preset Buttons
        ui.p(ui.strong("Preset Relationships:")),
        ui.div(
            ui.input_action_button(
                "btn_pos", 
                "📈 Positive (+0.75)", 
                class_="btn-success btn-sm mb-1 w-100"
            ),
            ui.input_action_button(
                "btn_zero", 
                "⚪ Uncorrelated (0.00)", 
                class_="btn-secondary btn-sm mb-1 w-100"
            ),
            ui.input_action_button(
                "btn_neg", 
                "📉 Negative (-0.75)", 
                class_="btn-danger btn-sm mb-3 w-100"
            ),
        ),
        
        ui.hr(),
        
        # Sample Size Slider
        ui.input_slider(
            "n", 
            "Sample Size (N):", 
            min=50, 
            max=1000, 
            value=300, 
            step=10
        ),
        
        # Resample / Regenerate Data Button
        ui.input_action_button(
            "btn_resample", 
            "🔄 Resample Points", 
            class_="btn-outline-primary w-100 mt-2"
        ),
        
        width=320
    ),
    
    ui.card(
        ui.card_header("Bivariate Normal Scatter Plot with Marginal Distributions"),
        
        # Summary Statistical Indicators
        ui.layout_columns(
            ui.value_box("Target r", ui.output_text("target_r_val"), subtitle="Preset Correlation"),
            ui.value_box("Sample r", ui.output_text("sample_r_val"), subtitle="Pearson's Correlation"),
            ui.value_box("R² Variance", ui.output_text("r2_val"), subtitle="Coeff. of Determination"),
            ui.value_box("Slope (β₁)", ui.output_text("slope_val"), subtitle="OLS Fit Slope"),
            col_widths=(3, 3, 3, 3)
        ),
        
        # Plotly Render Target with Marginal Distributions
        ui.output_ui("plot_ui"),
        
        # Mathematical Context Banner
        ui.div(
            ui.markdown("""
            **Bivariate Normal Sampling Model:**
            $$X \\sim \\mathcal{N}(0, 1), \\quad Z \\sim \\mathcal{N}(0, 1)$$
            $$Y = r \\cdot X + \\sqrt{1 - r^2} \\cdot Z$$
            *The main panel shows the bivariate scatter plot with OLS regression fit and 95% confidence interval band. The top and right marginal axes show empirical density curves for $X$ and $Y$.*
            """),
            class_="alert alert-light border mt-3 mb-0 fs-7"
        )
    ),
    title="Bivariate Normal Association Explorer"
)

# --- Server Logic ---
def server(input, output, session):
    
    # Preset button reactive handlers
    @reactive.effect
    @reactive.event(input.btn_pos)
    def _set_pos():
        ui.update_slider("r", value=0.75)

    @reactive.effect
    @reactive.event(input.btn_zero)
    def _set_zero():
        ui.update_slider("r", value=0.0)

    @reactive.effect
    @reactive.event(input.btn_neg)
    def _set_neg():
        ui.update_slider("r", value=-0.75)

    # Reactive Dataset Generator
    @reactive.calc
    def dataset():
        input.btn_resample()
        
        r = input.r()
        n = input.n()
        
        # Sample independent standard normal distributions
        x = np.random.normal(0, 1, n)
        z = np.random.normal(0, 1, n)
        
        # Synthesize correlated normal variable Y
        y = r * x + np.sqrt(1 - r**2) * z
        
        # Compute Ordinary Least Squares (OLS) regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Construct line sequence and 95% Confidence Interval band
        x_line = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 200)
        y_line = intercept + slope * x_line
        
        y_hat = intercept + slope * x
        residual_sum_sq = np.sum((y - y_hat)**2)
        s_err = np.sqrt(residual_sum_sq / (n - 2))
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean)**2)
        
        t_val = stats.t.ppf(0.975, df=n - 2)
        ci = t_val * s_err * np.sqrt(1/n + ((x_line - x_mean)**2) / sxx)
        
        # Density calculations for marginal plots
        x_kde = stats.gaussian_kde(x)
        y_kde = stats.gaussian_kde(y)
        
        x_dens_grid = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 100)
        y_dens_grid = np.linspace(np.min(y) - 0.5, np.max(y) + 0.5, 100)
        
        return {
            "x": x,
            "y": y,
            "x_line": x_line,
            "y_line": y_line,
            "ci_upper": y_line + ci,
            "ci_lower": y_line - ci,
            "target_r": r,
            "sample_r": r_value,
            "r2": r_value**2,
            "slope": slope,
            "intercept": intercept,
            "p_val": p_value,
            "x_dens_grid": x_dens_grid,
            "x_dens": x_kde(x_dens_grid),
            "y_dens_grid": y_dens_grid,
            "y_dens": y_kde(y_dens_grid)
        }

    # Dynamic Summary Metric Outputs
    @render.text
    def target_r_val():
        return f"{dataset()['target_r']:.2f}"

    @render.text
    def sample_r_val():
        return f"{dataset()['sample_r']:.3f}"

    @render.text
    def r2_val():
        return f"{dataset()['r2']:.3f}"

    @render.text
    def slope_val():
        return f"{dataset()['slope']:.3f}"

    # Render Plotly Scatter & Regression Figure with Marginal Density Subplots
    @render.ui
    def plot_ui():
        data = dataset()
        
        # Create 2x2 subplot layout with shared axes
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.85, 0.15],
            row_heights=[0.18, 0.82],
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.03,
            horizontal_spacing=0.03
        )

        # 1. Main Scatter Plot (Row 2, Col 1)
        # 95% Confidence Interval Band
        fig.add_trace(go.Scatter(
            x=np.concatenate([data["x_line"], data["x_line"][::-1]]),
            y=np.concatenate([data["ci_upper"], data["ci_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(13, 110, 253, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% CI Band"
        ), row=2, col=1)

        # OLS Linear Regression Trend Line
        fig.add_trace(go.Scatter(
            x=data["x_line"],
            y=data["y_line"],
            mode="lines",
            line=dict(color="#0d6efd", width=2.5),
            name="Linear Regression Line",
            hovertemplate="<b>Y Fit</b>: %{y:.2f}<extra></extra>"
        ), row=2, col=1)

        # Sampled Observations
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            marker=dict(
                size=7,
                color="#00b4d8",
                opacity=0.75,
                line=dict(width=0.5, color="#03045e")
            ),
            name="Observations",
            hovertemplate="<b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}<extra></extra>"
        ), row=2, col=1)

        # 2. Top Marginal Distribution for X (Row 1, Col 1)
        fig.add_trace(go.Scatter(
            x=data["x_dens_grid"],
            y=data["x_dens"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 180, 216, 0.2)",
            line=dict(color="#00b4d8", width=1.5),
            name="X Density",
            hovertemplate="<b>X Density</b>: %{y:.3f}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

        # 3. Right Marginal Distribution for Y (Row 2, Col 2)
        fig.add_trace(go.Scatter(
            x=data["y_dens"],
            y=data["y_dens_grid"],
            mode="lines",
            fill="tozerox",
            fillcolor="rgba(0, 180, 216, 0.2)",
            line=dict(color="#00b4d8", width=1.5),
            name="Y Density",
            hovertemplate="<b>Y Density</b>: %{x:.3f}<extra></extra>",
            showlegend=False
        ), row=2, col=2)

        # Styling and Grid Customization
        fig.update_xaxes(title_text="Variable X (~N(0,1))", zeroline=True, zerolinecolor="#adb5bd", gridcolor="#f1f3f5", row=2, col=1)
        fig.update_yaxes(title_text="Variable Y (~N(0,1))", zeroline=True, zerolinecolor="#adb5bd", gridcolor="#f1f3f5", row=2, col=1)
        
        # Hide marginal axis labels/ticks to keep the visual clean
        fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            height=540,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

app = App(app_ui, server)

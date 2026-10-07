import numpy as np
from scipy import stats
import plotly.graph_objects as go
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
        ui.card_header("Bivariate Normal Scatter Plot & Linear Regression"),
        
        # Summary Statistical Indicators
        ui.layout_columns(
            ui.value_box("Target r", ui.output_text("target_r_val"), subtitle="Preset Correlation"),
            ui.value_box("Sample r", ui.output_text("sample_r_val"), subtitle="Pearson's Correlation"),
            ui.value_box("R² Variance", ui.output_text("r2_val"), subtitle="Coeff. of Determination"),
            ui.value_box("Slope (β₁)", ui.output_text("slope_val"), subtitle="OLS Fit Slope"),
            col_widths=(3, 3, 3, 3)
        ),
        
        # Plotly Render Target
        ui.output_ui("plot_ui"),
        
        # Mathematical Context Banner
        ui.div(
            ui.markdown("""
            **Bivariate Normal Sampling Model:**
            $$X \\sim \\mathcal{N}(0, 1), \\quad Z \\sim \\mathcal{N}(0, 1)$$
            $$Y = r \\cdot X + \\sqrt{1 - r^2} \\cdot Z$$
            *This transformation ensures $Y$ follows a standard normal distribution with $E[Y]=0, \\text{Var}(Y)=1$, and theoretical correlation $\\text{Corr}(X,Y) = r$.*
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
        # Re-evaluates on resample click or input slider change
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
            "p_val": p_value
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

    # Render Plotly Scatter & Regression Figure
    @render.ui
    def plot_ui():
        data = dataset()
        
        fig = go.Figure()

        # 95% Confidence Interval Band
        fig.add_trace(go.Scatter(
            x=np.concatenate([data["x_line"], data["x_line"][::-1]]),
            y=np.concatenate([data["ci_upper"], data["ci_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(13, 110, 253, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% CI Band"
        ))

        # OLS Linear Regression Trend Line
        fig.add_trace(go.Scatter(
            x=data["x_line"],
            y=data["y_line"],
            mode="lines",
            line=dict(color="#0d6efd", width=2.5),
            name="Linear Regression Line",
            hovertemplate="<b>Y Fit</b>: %{y:.2f}<extra></extra>"
        ))

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
        ))

        # Crosshairs & Axis Layout
        fig.update_layout(
            xaxis=dict(title="Variable X (~N(0,1))", zeroline=True, zerolinecolor="#adb5bd", gridcolor="#f1f3f5"),
            yaxis=dict(title="Variable Y (~N(0,1))", zeroline=True, zerolinecolor="#adb5bd", gridcolor="#f1f3f5"),
            template="plotly_white",
            margin=dict(l=40, r=20, t=20, b=40),
            height=480,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

app = App(app_ui, server)

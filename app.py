import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

# Set Seaborn theme aesthetics
sns.set_theme(style="whitegrid")

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Control Panel"),
        ui.hr(),
        ui.input_slider(
            "corr",
            "Correlation Coefficient (r):",
            min=-0.99,
            max=0.99,
            value=0.75,
            step=0.01,
        ),
        ui.input_slider(
            "n_obs",
            "Number of Observations:",
            min=50,
            max=1000,
            value=250,
            step=50,
        ),
        ui.input_action_button(
            "btn_pos",
            "Set Positive (+0.75)",
            class_="btn-primary w-100 mb-2",
        ),
        ui.input_action_button(
            "btn_neg",
            "Set Negative (-0.75)",
            class_="btn-warning w-100 mb-2",
        ),
        ui.input_action_button(
            "btn_resample",
            "Resample Data",
            class_="btn-secondary w-100",
        ),
        ui.hr(),
        ui.markdown(
            """
            **How it works:**  
            Data is generated using a **bivariate normal distribution** with target covariance matrix:  
            **Σ = [[1, r], [r, 1]]**
            """
        ),
        width=300,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Scatter Plot with Linear Regression Trend Line"),
            ui.output_plot("scatter_plot"),
        ),
        ui.card(
            ui.card_header("Summary Statistics & Metrics"),
            ui.output_ui("stats_table"),
            ui.markdown(
                """
                ### Key Takeaways
                - **Correlation (r):** Controls the strength and direction of association.
                - **Trend Line:** Fits Ordinary Least Squares (OLS) line **Ŷ = β₀ + β₁X**.
                - **R² Score:** Proportion of variance in Y explained by X.
                """
            ),
        ),
        col_widths=[7, 5],
    ),
    title="Interactive Bivariate Normal Scatter Plot",
    fillable=False,
)


def server(input: Inputs, output: Outputs, session: Session):

    # Reactive event listeners for correlation preset buttons
    @reactive.effect
    @reactive.event(input.btn_pos)
    def _set_positive():
        ui.update_slider("corr", value=0.75)

    @reactive.effect
    @reactive.event(input.btn_neg)
    def _set_negative():
        ui.update_slider("corr", value=-0.75)

    # Reactive calculation to generate data
    @reactive.calc
    def dataset():
        input.btn_resample()

        r = input.corr()
        n = input.n_obs()

        cov_matrix = np.array([[1.0, r], [r, 1.0]])
        mean = [0.0, 0.0]
        data = np.random.multivariate_normal(mean, cov_matrix, size=n)

        return pd.DataFrame(data, columns=["X", "Y"])

    @render.plot
    def scatter_plot():
        df = dataset()
        r = input.corr()

        fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

        sns.regplot(
            data=df,
            x="X",
            y="Y",
            ax=ax,
            color="#2563eb" if r >= 0 else "#dc2626",
            scatter_kws={"alpha": 0.6, "s": 30, "edgecolor": "none"},
            line_kws={"color": "#0f172a", "linewidth": 2.5, "label": "OLS Trend Line"},
        )

        slope, intercept, r_val, p_val, std_err = stats.linregress(df["X"], df["Y"])

        ax.set_title(
            f"Target r = {r:.2f} | Sample r = {r_val:.2f}\n"
            f"Fit: Y = {intercept:.2f} + {slope:.2f}X",
            fontsize=11,
            pad=10,
        )
        ax.set_xlabel("Variable X ~ N(0, 1)", fontsize=10)
        ax.set_ylabel("Variable Y ~ N(0, 1)", fontsize=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend(loc="upper left")
        plt.tight_layout()

        return fig

    @render.ui
    def stats_table():
        df = dataset()
        slope, intercept, r_val, p_val, std_err = stats.linregress(df["X"], df["Y"])

        stats_data = [
            ("Sample Size (N)", f"{len(df)}"),
            ("Target Correlation (r)", f"{input.corr():.2f}"),
            ("Sample Correlation (r)", f"{r_val:.4f}"),
            ("R² Score", f"{r_val**2:.4f}"),
            ("Slope (β₁)", f"{slope:.4f}"),
            ("Intercept (β₀)", f"{intercept:.4f}"),
            ("p-value", f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"),
        ]

        rows = [
            ui.tags.tr(ui.tags.td(metric), ui.tags.td(value))
            for metric, value in stats_data
        ]

        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(ui.tags.th("Metric"), ui.tags.th("Value"))
            ),
            ui.tags.tbody(*rows),
            class_="table table-sm table-striped table-hover",
        )


app = App(app_ui, server)

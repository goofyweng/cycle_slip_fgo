import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def draw_vertical_line(ax, z: int, color, label):
    """
    Draw a vertical line at value z in the horizontal axis
    """
    # Add a vertical line at a specific value on the x-axis (e.g., x=6)
    # Plot a vertical line at x = 5
    ax.axvline(x=z, color=color, linestyle="--", linewidth=2, label=label)


def plot_non_central_chi2(ax, dof, nc, xlim):
    # Generate x values for the plot
    x = np.linspace(0, xlim, 1000)

    # Calculate the PDF for the non-central chi-squared distribution
    pdf = stats.ncx2.pdf(x, dof, nc)

    # Plot the distribution on the specified axis
    ax.plot(x, pdf, label=f"Non-central Chi-Squared\n(df={dof}, nc={nc:.2f})")

    # Customize the plot
    ax.set_title("Non-central Central Chi-Squared Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")

"""Optional plotting helper shared by the examples (matplotlib is not a dependency)."""

from pathlib import Path


def figure(name):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    fig = plt.figure(figsize=(9, 4))
    return fig, Path(__file__).with_name(f"{name}.png")

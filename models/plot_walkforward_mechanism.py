"""
Schematic diagram of the two walk-forward validation schemes used across
this project:
  1. Season-level walk-forward (Poisson_Covariates_Bivariate.py, RF.py,
     XGBoost.py, etc.): train on WINDOW prior full seasons, test on the
     whole next season in one shot, slide forward by one season per fold.
  2. Intra-season cumulative sliding (RF_intraseason_walkforward.py,
     Poisson_Bivariate_intraseason_walkforward.py, FFNN.py): same WINDOW
     prior-seasons base, but the test season itself is split into N_CHUNKS
     chronological pieces, and training grows by one chunk each step within
     the season instead of testing the whole season at once.

Saved to images/walkforward_mechanism.png.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEASON_LEVEL_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\images\walkforward_scheme.png"
INTRASEASON_PATH = r"C:\Users\misog\SCHOOL\Summer project\ML-football-odds\images\walkforward_intraseason_scheme.png"

TRAIN_COLOR = "#4C72B0"
ELAPSED_COLOR = "#DD8452"
TEST_COLOR = "#C44E52"
FUTURE_COLOR = "#E8E8E8"
WINDOW = 3
N_CHUNKS = 5
SEASONS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def draw_block(ax, x0, x1, y, color, edgecolor="white", hatch=None):
    ax.add_patch(mpatches.Rectangle(
        (x0, y), x1 - x0, 0.8, facecolor=color, edgecolor=edgecolor, linewidth=1.5, hatch=hatch
    ))


def panel_season_level(ax):
    n_seasons = len(SEASONS)
    folds = [(i - WINDOW, i, i) for i in range(WINDOW, n_seasons)]  # (train_start, train_end_excl, test_idx)

    for row, (t0, t1, test_idx) in enumerate(folds):
        y = (len(folds) - 1 - row) * 1.2
        for s in range(t0, t1):
            draw_block(ax, s, s + 1, y, TRAIN_COLOR)
        draw_block(ax, test_idx, test_idx + 1, y, TEST_COLOR)
        for s in range(t1, test_idx):
            pass
        ax.text(-0.3, y + 0.4, f"Fold {row + 1}", ha="right", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(-1.6, n_seasons + 0.3)
    ax.set_ylim(-0.3, len(folds) * 1.2)
    ax.set_xticks([i + 0.5 for i in range(n_seasons)])
    ax.set_xticklabels(SEASONS)
    ax.set_yticks([])
    ax.set_title(f"WINDOW={WINDOW}: train on {WINDOW} prior full seasons -> test the whole next season", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_intraseason(ax):
    prior_width = WINDOW  # collapse the WINDOW prior seasons into one block of this width
    season_start = prior_width + 0.4  # gap between prior-seasons block and the test season
    chunk_width = 1.0

    for c_idx in range(N_CHUNKS):
        y = (N_CHUNKS - 1 - c_idx) * 1.2

        # Fixed prior-WINDOW-seasons training base
        draw_block(ax, 0, prior_width, y, TRAIN_COLOR)

        # Elapsed chunks of the current season already folded into training
        for c in range(c_idx):
            x0 = season_start + c * chunk_width
            draw_block(ax, x0, x0 + chunk_width, y, ELAPSED_COLOR)

        # The chunk being tested right now
        x0 = season_start + c_idx * chunk_width
        draw_block(ax, x0, x0 + chunk_width, y, TEST_COLOR)

        # Remaining chunks not yet reached this fold
        for c in range(c_idx + 1, N_CHUNKS):
            x0 = season_start + c * chunk_width
            draw_block(ax, x0, x0 + chunk_width, y, FUTURE_COLOR, edgecolor="#bbbbbb")

        ax.text(-0.3, y + 0.4, f"Chunk {c_idx}", ha="right", va="center", fontsize=10, fontweight="bold")

    ax.axvline(prior_width, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(prior_width / 2, N_CHUNKS * 1.2 + 0.05, "Prior WINDOW seasons\n(fixed)", ha="center", fontsize=9)
    ax.text(season_start + N_CHUNKS * chunk_width / 2, N_CHUNKS * 1.2 + 0.05, "Current (test) season, split into chunks", ha="center", fontsize=9)

    ax.set_xlim(-1.6, season_start + N_CHUNKS * chunk_width + 0.3)
    ax.set_ylim(-0.3, N_CHUNKS * 1.2 + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"N_CHUNKS={N_CHUNKS}: training also grows chunk-by-chunk WITHIN the test season", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    # --- Figure 1: season-level walk-forward ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    panel_season_level(ax1)
    legend1 = [
        mpatches.Patch(facecolor=TRAIN_COLOR, label="Training (prior full seasons)"),
        mpatches.Patch(facecolor=TEST_COLOR, label="Test (evaluated this fold)"),
    ]
    fig1.legend(handles=legend1, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig1.suptitle("Season-level walk-forward validation", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(SEASON_LEVEL_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {SEASON_LEVEL_PATH}")
    plt.close(fig1)

    # --- Figure 2: intra-season cumulative sliding ---
    fig2, ax2 = plt.subplots(figsize=(10, 6.5))
    panel_intraseason(ax2)
    legend2 = [
        mpatches.Patch(facecolor=TRAIN_COLOR, label="Training (prior full seasons)"),
        mpatches.Patch(facecolor=ELAPSED_COLOR, label="Training (elapsed chunks of test season)"),
        mpatches.Patch(facecolor=TEST_COLOR, label="Test (evaluated this step)"),
        mpatches.Patch(facecolor=FUTURE_COLOR, edgecolor="#bbbbbb", label="Not yet reached this fold"),
    ]
    fig2.legend(handles=legend2, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig2.suptitle("Intra-season cumulative sliding walk-forward", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.09, 1, 0.95])
    plt.savefig(INTRASEASON_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {INTRASEASON_PATH}")
    plt.close(fig2)


if __name__ == "__main__":
    main()

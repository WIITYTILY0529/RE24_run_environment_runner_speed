"""Build the RE24 design with wOBA and runner speed effects that vary with team RPG."""

import numpy as np

# Column layout of design(). The HTML explorer reads fit.npz beta by these positions.
STATE = slice(0, 24)        # base-out state intercepts (state = 8 * outs + bases)
WOBA = slice(24, 48)        # state x wOBA in .010 units
RPG = slice(48, 72)         # state x centered team RPG
RUNNER = slice(72, 75)      # centered measured speed on first, second, third
MISSING = slice(75, 78)     # occupied base with unknown speed
WOBA_RPG = slice(78, 102)   # state x wOBA x centered team RPG
RUNNER_RPG = 102            # centered team RPG x summed centered runner speed
N_COLUMNS = 103


def design(rows, rpg_reference, speed_reference):
    """Build 103 columns with RPG in runs per team per game.

    Columns 48:72 contain centered team RPG by state. Column 102 is
    centered team RPG times the sum of centered measured runner speeds.
    Its single coefficient modifies the speed slopes for all three bases.
    """
    state = np.eye(24)[(8 * rows.outs + rows.bases).to_numpy(dtype=int)]
    rpg = rows.runs_per_game.to_numpy() / 2 - rpg_reference
    woba = 100 * rows.woba.to_numpy()
    speed = rows[["speed_1", "speed_2", "speed_3"]].to_numpy(dtype=float)
    occupied = rows[["runner_1", "runner_2", "runner_3"]].notna().to_numpy()
    missing = occupied & np.isnan(speed)
    runner = np.where(occupied & ~missing, speed - speed_reference, 0)
    return np.column_stack((
        state,
        state * woba[:, None],
        state * rpg[:, None],
        runner,
        missing,
        state * (woba * rpg)[:, None],
        rpg * runner.sum(axis=1),
    ))

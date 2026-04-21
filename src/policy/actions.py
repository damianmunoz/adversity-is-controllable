"""
Action space for the execution policy.

Three actions model how aggressively we interact with the order book:

  WAIT       — do nothing this tick. Appropriate when state is uncertain or
               market pressure is low. Zero cost, zero fill.

  PASSIVE    — post a limit order at the best bid/ask. We earn the spread
               but risk not getting filled if the market moves away.

  AGGRESSIVE — cross the spread with a market order. Guaranteed fill but
               we pay the spread as cost (adverse execution price).

These map directly to the simulation in Step 6: WAIT gives no fill,
PASSIVE gives a fill only if the book doesn't move against us,
AGGRESSIVE always fills at the current best opposite price.
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    WAIT       = "WAIT"
    PASSIVE    = "PASSIVE"
    AGGRESSIVE = "AGGRESSIVE"

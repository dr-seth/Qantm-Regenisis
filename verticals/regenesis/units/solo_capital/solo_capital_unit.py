"""GovernedUnit for solo_capital domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class SoloCapitalUnit(GovernedUnit):
    """Governed unit for regenesis.solo_capital domain."""

    DOMAIN = "regenesis.solo_capital"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="SoloCapitalUnit",
            autonomy_level=autonomy_level,
        )

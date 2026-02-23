"""GovernedUnit for energy domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class EnergyUnit(GovernedUnit):
    """Governed unit for regenesis.energy domain."""

    DOMAIN = "regenesis.energy"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="EnergyUnit",
            autonomy_level=autonomy_level,
        )

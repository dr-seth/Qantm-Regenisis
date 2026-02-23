"""GovernedUnit for zoning domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class ZoningUnit(GovernedUnit):
    """Governed unit for regenesis.zoning domain."""

    DOMAIN = "regenesis.zoning"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="ZoningUnit",
            autonomy_level=autonomy_level,
        )

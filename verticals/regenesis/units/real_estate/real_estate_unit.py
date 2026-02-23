"""GovernedUnit for real_estate domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class RealEstateUnit(GovernedUnit):
    """Governed unit for regenesis.real_estate domain."""

    DOMAIN = "regenesis.real_estate"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="RealEstateUnit",
            autonomy_level=autonomy_level,
        )

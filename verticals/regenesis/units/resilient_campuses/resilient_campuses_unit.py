"""GovernedUnit for resilient_campuses domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class ResilientCampusesUnit(GovernedUnit):
    """Governed unit for regenesis.resilient_campuses domain."""

    DOMAIN = "regenesis.resilient_campuses"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="ResilientCampusesUnit",
            autonomy_level=autonomy_level,
        )

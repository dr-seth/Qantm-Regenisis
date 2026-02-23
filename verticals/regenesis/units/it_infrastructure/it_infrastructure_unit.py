"""GovernedUnit for it_infrastructure domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class ItInfrastructureUnit(GovernedUnit):
    """Governed unit for regenesis.it_infrastructure domain."""

    DOMAIN = "regenesis.it_infrastructure"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="ItInfrastructureUnit",
            autonomy_level=autonomy_level,
        )

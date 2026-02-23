"""GovernedUnit for autonomous_intelligence domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class AutonomousIntelligenceUnit(GovernedUnit):
    """Governed unit for regenesis.autonomous_intelligence domain."""

    DOMAIN = "regenesis.autonomous_intelligence"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="AutonomousIntelligenceUnit",
            autonomy_level=autonomy_level,
        )

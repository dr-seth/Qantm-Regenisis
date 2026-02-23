"""GovernedUnit for compliance domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class ComplianceUnit(GovernedUnit):
    """Governed unit for regenesis.compliance domain."""

    DOMAIN = "regenesis.compliance"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="ComplianceUnit",
            autonomy_level=autonomy_level,
        )

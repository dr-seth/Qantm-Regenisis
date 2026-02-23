"""GovernedUnit for distressed_assets domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class DistressedAssetsUnit(GovernedUnit):
    """Governed unit for regenesis.distressed_assets domain."""

    DOMAIN = "regenesis.distressed_assets"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="DistressedAssetsUnit",
            autonomy_level=autonomy_level,
        )

"""GovernedUnit for site_topology domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class SiteTopologyUnit(GovernedUnit):
    """Governed unit for regenesis.site_topology domain."""

    DOMAIN = "regenesis.site_topology"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="SiteTopologyUnit",
            autonomy_level=autonomy_level,
        )

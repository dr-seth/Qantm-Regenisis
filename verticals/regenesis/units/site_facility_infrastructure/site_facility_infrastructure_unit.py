"""GovernedUnit for site_facility_infrastructure domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class SiteFacilityInfrastructureUnit(GovernedUnit):
    """Governed unit for regenesis.site_facility_infrastructure domain."""

    DOMAIN = "regenesis.site_facility_infrastructure"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="SiteFacilityInfrastructureUnit",
            autonomy_level=autonomy_level,
        )

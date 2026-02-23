"""GovernedUnit for weather_climate domain in ReGenesis - Resilient Community Conversion."""

from packages.core.governed_unit import GovernedUnit
from packages.core.types import AutonomyLevel


class WeatherClimateUnit(GovernedUnit):
    """Governed unit for regenesis.weather_climate domain."""

    DOMAIN = "regenesis.weather_climate"

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.PROPOSE_ONLY):
        super().__init__(
            name="WeatherClimateUnit",
            autonomy_level=autonomy_level,
        )

ENVIRONMENT_TYPE_FLAT = "flat"
ENVIRONMENT_TYPE_MOUNTAINOUS = "mountainous"
CLIMATE_TEMPERATE = "temperate"
CLIMATE_DRY = "dry"

FEATURE_PROBABILITIES ={
            "linear": 0.25,
            "field": 0.14,
            "structure": 0.13,
            "road": 0.13,
            "drainage": 0.12,
            "water": 0.08,
            "brush": 0.02,
            "scrub": 0.03,
            "woodland": 0.07,
            "rock": 0.04,
        }

RADIUS_FLAT_TEMPERATE = [0.6, 1.8,3.2,9.9]
RADIUS_FLAT_DRY = [1.3, 2.1,6.6,13.1]
RADIUS_MOUNTAINOUS_TEMPERATE = [1.1, 3.1,5.8,18.3]
RADIUS_MOUNTAINOUS_DRY = [1.6, 3.2,6.5,19.3]



def get_environment_radius(environment_type, environment_climate):
    """
    Get the radius values based on the environment type and climate.

    Args:
        environment_type (str): The type of environment (flat or mountainous).
        climate (str): The climate type (temperate or dry).

    Returns:
        list: A list of radius values.
    """
    if environment_type == ENVIRONMENT_TYPE_FLAT:
        if environment_climate == CLIMATE_TEMPERATE:
            return RADIUS_FLAT_TEMPERATE
        if environment_climate == CLIMATE_DRY:
            return RADIUS_FLAT_DRY
    elif environment_type == ENVIRONMENT_TYPE_MOUNTAINOUS:
        if environment_climate == CLIMATE_TEMPERATE:
            return RADIUS_MOUNTAINOUS_TEMPERATE
        if environment_climate == CLIMATE_DRY:
            return RADIUS_MOUNTAINOUS_DRY
    return []

def get_environment_radius_by_size(environment_type, environment_climate, size):
    """
    Get the radius values based on the environment type and climate.

    Args:
        environment_type (str): The type of environment (flat or mountainous).
        climate (str): The climate type (temperate or dry).

    Returns:
        list: A list of radius values.
    """
    if size == "small":
        index = 0
    elif size == "medium":
        index = 1
    elif size == "large":
        index = 2
    elif size == "xlarge":
        index = 3
    else:
        raise ValueError(f"Invalid size: {size}. Expected one of ['small', 'medium', 'large', 'extra_large'].")
    return get_environment_radius(environment_type,environment_climate)[index]

def get_available_sizes() -> list[str]:
    """Returns a list of all predefined size names."""
    # This assumes a standard set of sizes. If sizes can vary, this might need adjustment.
    return ["small", "medium", "large", "xlarge"]

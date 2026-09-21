# sarenv/utils/geo.py
"""
Geographic utility functions.
"""
def get_utm_epsg(lon: float, lat: float) -> str:
    """Calculates the appropriate UTM zone EPSG code for a given point."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{326 if lat >= 0 else 327}{zone}"

# Corrected global helper function
def world_to_image(x, y, meters_per_bin, minx, miny, buffer_val):
    # x and y can be NumPy arrays
    x_img = (x - minx + buffer_val) / meters_per_bin
    y_img = (y - miny + buffer_val) / meters_per_bin
    # Convert to int array, then return.
    # If x_img, y_img are single scalars, astype(int) also works.
    return x_img.astype(int), y_img.astype(int)


# This function is not strictly needed if Environment.world_to_image calls the global one directly
# but kept for structural consistency if preferred.
def image_to_world(x_img, y_img, meters_per_bin, minx, miny, buffer_val):
    x_world = x_img * meters_per_bin + minx - buffer_val
    y_world = y_img * meters_per_bin + miny - buffer_val
    return x_world, y_world


# sarenv/io/osm_query.py
from pathlib import Path
import osmnx as ox
import shapely
from geojson import Feature, FeatureCollection, dump # Ensure geojson is in requirements

# Relative imports
from ..utils.logging_setup import get_logger
from ..core.geometries import GeoPolygon # Assuming GeoPolygon is defined in core.geometries

log = get_logger()

def query_features(area_geopolygon: GeoPolygon, tags_to_query: dict) -> dict | None:
    """
    Queries features from OpenStreetMap within the given area_geopolygon using specified tags.

    Parameters:
    -----------
    area_geopolygon: GeoPolygon object representing the query area. Must be in WGS84 (EPSG:4326).
    tags_to_query: dict, OSM tags to query (e.g., {"building": True}). This function expects
                     that this dict represents a single feature type for querying, where keys
                     are OSM tags and values are tag values.
                     If you need to query multiple types, call this function for each.

    Returns:
    --------
    dict | None: A dictionary where keys are the top-level keys from tags_to_query (representing categories)
                  and values are Shapely geometries (or MultiGeometries) of the queried features,
                  intersected with the area_geopolygon. Geometries are in WGS84.
                  Returns None if an error occurs during querying or if the area_geopolygon CRS is not WGS84.
    """
    if area_geopolygon.crs != "EPSG:4326" and area_geopolygon.crs != "WGS84": # OSMnx expects WGS84
        log.error(f"The area_geopolygon must use WGS84 (EPSG:4326) CRS, but found {area_geopolygon.crs}.")

        raise ValueError("Query area must be in WGS84 (EPSG:4326) CRS.")


    try:
        # ox.features_from_polygon expects the polygon geometry itself
        raw_osm_geometries_gdf = ox.features_from_polygon(area_geopolygon.get_geometry(), tags=tags_to_query)
    except Exception as e:
        log.warning("An error occurred while querying features from OSM: %s", str(e))
        return None

    if raw_osm_geometries_gdf.empty:
        log.info("No features found from OSM for tags: %s", str(tags_to_query))
        return {tag_key: None for tag_key in tags_to_query.keys()} # Return None for each requested tag key

    consolidated_geometry = raw_osm_geometries_gdf.geometry.unary_union
    if consolidated_geometry.is_empty:
        log.info("Consolidated geometry is empty after OSM query for tags: %s", str(tags_to_query))
        return None # No specific features matched or consolidated to empty

    # Intersect with the precise query boundary
    final_features_geom = shapely.intersection(area_geopolygon.get_geometry(), consolidated_geometry)

    if final_features_geom.is_empty:
        log.info("Intersection with query area resulted in empty geometry for tags: %s", str(tags_to_query))
        return None

    log.info("Successfully queried and processed features for tags: %s. Result type: %s", str(tags_to_query), type(final_features_geom))

    final_dict_for_env = {}
    if not raw_osm_geometries_gdf.empty:
        for osm_key in tags_to_query.keys(): # e.g., "building", "highway"
            if osm_key in raw_osm_geometries_gdf.columns:
                col_geoms = raw_osm_geometries_gdf[raw_osm_geometries_gdf[osm_key].notna()].geometry
                if not col_geoms.empty:
                    intersected_col_geom = shapely.intersection(area_geopolygon.get_geometry(), col_geoms.unary_union)
                    if not intersected_col_geom.is_empty:
                        final_dict_for_env[osm_key] = intersected_col_geom
    if not final_dict_for_env:
        log.info(f"Query for tags {tags_to_query} resulted in no specific features after filtering by individual tags.")
        return None
    return final_dict_for_env



def export_as_geojson(boundary_geom: shapely.Polygon, obstacles_list: list[shapely.Polygon], features_list: list[shapely.LineString], crs_str: str, output_filepath: str | Path):
    """Exports boundary, obstacles, and features (tasks) to a GeoJSON file."""
    if not isinstance(boundary_geom, shapely.Polygon):
        raise TypeError("boundary_geom must be a Shapely Polygon.")
    if not all(isinstance(obs, shapely.Polygon) for obs in obstacles_list):
        raise TypeError("All items in obstacles_list must be Shapely Polygons.")
    if not all(isinstance(feat, shapely.LineString) for feat in features_list):
        raise TypeError("All items in features_list must be Shapely LineStrings.")

    boundary_feature = Feature(geometry=boundary_geom, id="boundary", properties={"name":"Boundary"})
    
    # Consolidate obstacles into a MultiPolygon if multiple, or keep as Polygon if single
    obstacles_geom = None
    if obstacles_list:
        if len(obstacles_list) == 1:
            obstacles_geom = obstacles_list[0]
        else:
            obstacles_geom = shapely.MultiPolygon(obstacles_list)
    obstacles_feature = Feature(geometry=obstacles_geom, id="obstacles", properties={"name":"Obstacles"}) if obstacles_geom else None

    # Consolidate features into a MultiLineString
    tasks_geom = None
    if features_list:
        if len(features_list) == 1:
            tasks_geom = features_list[0]
        else:
            tasks_geom = shapely.MultiLineString(features_list)
    task_feature = Feature(geometry=tasks_geom, id="tasks", properties={"name":"Tasks"}) if tasks_geom else None

    all_features_for_collection = [boundary_feature]
    if obstacles_feature: all_features_for_collection.append(obstacles_feature)
    if task_feature: all_features_for_collection.append(task_feature)

    # GeoJSON spec suggests CRS is often omitted at FeatureCollection level if all features are WGS84
    # And individual features don't usually carry CRS if the collection implies it.
    # For simplicity here, let's assume WGS84 is implied or features are already in desired output CRS.
    # The `crs_str` could be used to set a "crs" member at the FeatureCollection level if needed,
    # but this is often debated in GeoJSON best practices if it's not the default WGS84.
    feature_collection_dict = FeatureCollection(all_features_for_collection, properties={"crs_comment": f"Geometries assumed to be in CRS: {crs_str}"})


    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    with output_path.open("w") as geojson_file:
        dump(feature_collection_dict, geojson_file, indent=4)
    log.info(f"Exported data to GeoJSON: {output_path}")
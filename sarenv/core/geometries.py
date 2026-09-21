# sarenv/core/geometries.py
import math
import geojson
import pyproj
import shapely
import shapely.plotting as shplt
from shapely.geometry.polygon import orient
import random

# Use relative import for logging_setup
from ..utils.logging_setup import get_logger

log = get_logger()


class GeoData:
    def __init__(self, geometry, crs="WGS84"):
        self.geometry = geometry
        self.crs = crs

    def set_crs(self, crs):
        if not isinstance(crs, str):
            msg = "New CRS must be a string."
            raise ValueError(msg)

        if crs != self.crs:
            # Apply the transformer to the geometry
            self._convert_to_crs(crs)
        self.crs = crs
        return self

    def _convert_to_crs(self, crs):  # noqa: ARG002
        msg = "_convert_to_crs(crs) sould be implemented in the data classes!"
        raise NotImplementedError(msg)

    def is_geometry_of_type(self, geometry, expected_class):
        if expected_class and not isinstance(geometry, expected_class):
            msg = f"Geometry must be a {expected_class.__name__}."
            raise ValueError(msg)

    def get_geometry(self):
        return self.geometry

    def buffer(self, distance, quad_segs=1, cap_style="square", join_style="bevel"):
        self.geometry = self.geometry.buffer(distance, quad_segs, cap_style, join_style)
        return self

    def __str__(self):
        return f"Geometry in CRS: {self.crs}\nGeometry: {self.geometry}"

    def __geo_interface__(self):
        return self.geometry.__geo_interface__

    def to_geojson(self, id_val=None, name=None, properties=None): # renamed id to id_val
        if id_val is None:
            id_val = str(random.randint(0, 1000000000)) # Ensure id is string for geojson
        if name is None:
            name = str(id_val)
        if properties is None:
            properties = {}

        properties["crs"] = self.crs
        properties["name"] = name
        return geojson.Feature(id=id_val, geometry=self.geometry, properties=properties)


class GeoTrajectory(GeoData):
    def __init__(self, geometry, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.LineString)
        super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=True, color=None, linewidth=2, **kwargs):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        shplt.plot_line(self.geometry, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)

        converted_coords = [
            transformer.transform(x, y) for x, y in list(self.geometry.coords)
        ]
        self.geometry = shapely.LineString(converted_coords)


class GeoMultiTrajectory(GeoData):
    def __init__(
        self,
        geometry: (
            shapely.MultiLineString
            | list[shapely.LineString]
            | list[GeoTrajectory]
            | shapely.LineString
        ),
        crs="WGS84",
    ):
        # super().__init__(geometry, crs) # Call super after geometry is processed
        processed_geometry = None
        if isinstance(geometry, list):
            line_geoms = []
            for line_item in geometry:
                if isinstance(line_item, GeoTrajectory):
                     self.is_geometry_of_type(line_item.geometry, shapely.LineString)
                     line_geoms.append(line_item.geometry)
                elif isinstance(line_item, shapely.LineString):
                    self.is_geometry_of_type(line_item, shapely.LineString)
                    line_geoms.append(line_item)
                else:
                    msg = "List items must be shapely.LineString or GeoTrajectory instances."
                    raise ValueError(msg)
            processed_geometry = shapely.MultiLineString(line_geoms)
        elif isinstance(geometry, shapely.LineString):
            self.is_geometry_of_type(geometry, shapely.LineString)
            processed_geometry = shapely.MultiLineString([geometry])
        elif isinstance(geometry, GeoTrajectory): # Added this case
            self.is_geometry_of_type(geometry.geometry, shapely.LineString)
            processed_geometry = shapely.MultiLineString([geometry.geometry])
        elif isinstance(geometry, shapely.MultiLineString):
            self.is_geometry_of_type(geometry, shapely.MultiLineString)
            processed_geometry = geometry
        else:
            msg = f"Unsupported geometry type for GeoMultiTrajectory: {type(geometry)}"
            raise ValueError(msg)
        super().__init__(processed_geometry, crs)


    def plot(self, ax=None, add_points=False, color=None, linewidth=2, **kwargs):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        for line in self.geometry.geoms:
            shplt.plot_line(line, ax, add_points, color, linewidth, **kwargs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        converted_lines = []
        for line in self.geometry.geoms:
            converted_coords = [
                transformer.transform(x, y) for x, y in list(line.coords)
            ]
            converted_lines.append(shapely.LineString(converted_coords))
        self.geometry = shapely.MultiLineString(converted_lines)


class GeoPoint(GeoData):
    def __init__(self, geometry: shapely.Point, crs="WGS84"):
        self.is_geometry_of_type(geometry, shapely.Point)
        super().__init__(geometry, crs)

    def plot(self, ax=None, add_points=True, color=None, linewidth=2, **kwargs): # linewidth might not be typical for points
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )
        # `add_points` might be confusing here, `plot_points` usually plots points.
        # `linewidth` is not a standard arg for plot_points, marker size might be more relevant.
        shplt.plot_points(self.geometry, ax, color=color, **kwargs)


    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.geometry.x, self.geometry.y)
        self.geometry = shapely.Point(x, y)


class GeoPolygon(GeoData):
    def __init__(self, geometry: shapely.Polygon | shapely.LineString, crs="WGS84"):
        # Allow LineString to be converted to Polygon if it's a closed ring
        if isinstance(geometry, shapely.LineString):
            if geometry.is_ring:
                geometry = shapely.Polygon(geometry)
            else:
                msg = "LineString must be a closed ring to be converted to GeoPolygon."
                raise ValueError(msg)
        self.is_geometry_of_type(geometry, shapely.Polygon)
        super().__init__(geometry, crs)

    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        exterior = [
            transformer.transform(x, y) for x, y in self.geometry.exterior.coords
        ]
        interiors = [
            [transformer.transform(x, y) for x, y in interior.coords]
            for interior in self.geometry.interiors
        ]
        self.geometry = shapely.Polygon(exterior, interiors)

    def plot(
        self,
        ax=None,
        add_points=False,
        color=None, # A single color for both face and edge if others not set
        facecolor=None,
        edgecolor=None,
        linewidth=2,
        **kwargs,
    ):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )

        # Set default facecolor and edgecolor based on color if not provided
        final_facecolor = facecolor if facecolor is not None else (color if color is not None else 'blue') # Default to blue if nothing set
        final_edgecolor = edgecolor if edgecolor is not None else (color if color is not None else 'black')


        shplt.plot_polygon(
            polygon=self.geometry,
            ax=ax,
            add_points=add_points,
            facecolor=final_facecolor, # Use updated variable
            edgecolor=final_edgecolor, # Use updated variable
            linewidth=linewidth,
            **kwargs,
        )


class GeoMultiPolygon(GeoData):
    def __init__(self, geometry, crs="WGS84"):
        processed_geometry = None
        if isinstance(geometry, list):
            poly_geoms = []
            for geom_item in geometry:
                if isinstance(geom_item, GeoPolygon):
                    self.is_geometry_of_type(geom_item.geometry, shapely.Polygon)
                    poly_geoms.append(geom_item.geometry)
                elif isinstance(geom_item, shapely.Polygon):
                    self.is_geometry_of_type(geom_item, shapely.Polygon)
                    poly_geoms.append(geom_item)
                else:
                    msg = "List items must be shapely.Polygon or GeoPolygon instances."
                    raise ValueError(msg)
            processed_geometry = shapely.MultiPolygon(poly_geoms)
        elif isinstance(geometry, shapely.Polygon): # Allow single Polygon to be wrapped
            processed_geometry = shapely.MultiPolygon([geometry])
        elif isinstance(geometry, shapely.MultiPolygon):
            self.is_geometry_of_type(geometry, shapely.MultiPolygon)
            processed_geometry = geometry
        else:
            msg = f"Unsupported geometry type for GeoMultiPolygon: {type(geometry)}"
            raise ValueError(msg)
        super().__init__(processed_geometry, crs)


    def _convert_to_crs(self, crs):
        transformer = pyproj.Transformer.from_crs(self.crs, crs, always_xy=True)
        polygon_list = []
        for polygon in list(self.geometry.geoms):
            exterior = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
            interiors = [
                [transformer.transform(x, y) for x, y in interior.coords]
                for interior in polygon.interiors
            ]
            polygon_list.append(shapely.Polygon(exterior, interiors))
        self.geometry = shapely.MultiPolygon(polygon_list)

    def plot(
        self,
        ax=None,
        add_points=False,
        color=None, # Single color for all polygons if face/edge not distinct
        facecolor=None,
        edgecolor=None,
        linewidth=2,
        **kwargs,
    ):
        if self.crs == "WGS84":
            log.warning(
                "Plotting in WGS84 is not recomended as this distorts the geometry!"
            )

        final_facecolor = facecolor if facecolor is not None else (color if color is not None else 'blue')
        final_edgecolor = edgecolor if edgecolor is not None else (color if color is not None else 'black')

        for polygon in self.geometry.geoms: # Plot each polygon in the multipolygon
            shplt.plot_polygon(
                polygon=polygon, # Iterate and plot individual polygons
                ax=ax,
                add_points=add_points,
                facecolor=final_facecolor,
                edgecolor=final_edgecolor,
                linewidth=linewidth,
                **kwargs,
            )
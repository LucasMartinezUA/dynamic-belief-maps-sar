# sarenv/analytics/metrics.py
"""
Provides the PathEvaluator class to score coverage paths against various metrics.
"""
import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

class PathEvaluator:
    """
    Evaluates coverage paths against metrics like likelihood scores, time-discounted
    scores, and victim detection probabilities.
    """
    def __init__(self, heatmap: np.ndarray, extent: tuple, victims: gpd.GeoDataFrame, fov_deg: float, altitude: float, meters_per_bin: int):
        """
        Initializes the PathEvaluator.

        Args:
            heatmap (np.ndarray): The 2D numpy array representing the probability heatmap.
            extent (tuple): A tuple (minx, miny, maxx, maxy) defining the geographical bounds of the heatmap.
            victims (gpd.GeoDataFrame): A GeoDataFrame containing victim locations as points.
            fov_deg (float): The camera's field of view in degrees.
            altitude (float): The altitude of the drone in meters.
            meters_per_bin (int): The size of a heatmap cell in meters.
        """
        self.heatmap = heatmap
        self.extent = extent
        self.victims = victims
        self.detection_radius = altitude * np.tan(np.radians(fov_deg / 2))
        self.interpolation_resolution = int(np.ceil(meters_per_bin / 2))

        # Calculate actual heatmap cell size for proper cell_key calculation
        minx, miny, maxx, maxy = self.extent
        self.heatmap_cell_size_x = (maxx - minx) / heatmap.shape[1]
        self.heatmap_cell_size_y = (maxy - miny) / heatmap.shape[0]

        minx, miny, maxx, maxy = self.extent
        y_range = np.linspace(miny, maxy, heatmap.shape[0])
        x_range = np.linspace(minx, maxx, heatmap.shape[1])
        self.interpolator = RegularGridInterpolator((y_range, x_range), heatmap, bounds_error=False, fill_value=0)

    def calculate_all_metrics(self, paths: list[LineString], discount_factor) -> dict:
        """
        Calculates all metrics for a given list of paths using a view model that considers
        the detection radius to determine which grid cells are visible from each position.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.
            discount_factor (float, optional): The discount factor for time-discounted scores. Defaults to 0.999.

        Returns:
            dict: A dictionary containing all calculated metrics:
                  - 'total_likelihood_score'
                  - 'total_time_discounted_score'
                  - 'victim_detection_metrics'
                  - 'area_covered'
                  - 'total_path_length'
                  - 'cumulative_distances' (list of arrays)
                  - 'cumulative_likelihoods' (list of arrays)
                  - 'cumulative_time_discounted_scores' (list of arrays)
        """
        # Track globally observed cells using the view model
        globally_observed_cells = set()
        all_position_data = []
        path_metadata = []
        global_time_offset = 0

        # First pass: collect all positions and their visible cells
        for path_idx, path in enumerate(paths):
            if path.is_empty or path.length == 0:
                path_metadata.append({
                    'path_idx': path_idx,
                    'distances': np.array([0]),
                    'positions': [],
                    'global_distances': np.array([0]),
                    'visible_cells_per_position': []
                })
                continue

            num_points = int(np.ceil(path.length / self.interpolation_resolution)) + 1
            distances = np.linspace(0, path.length, num_points)
            points = [path.interpolate(d) for d in distances]
            positions = [(p.x, p.y) for p in points]

            # Global distances for time-discounted scores
            global_distances = distances + global_time_offset

            # Calculate visible cells for each position
            visible_cells_per_position = []
            for x, y in positions:
                visible_cells = self.get_visible_cells(x, y)
                visible_cells_per_position.append(visible_cells)

            # Store metadata for this path
            path_metadata.append({
                'path_idx': path_idx,
                'distances': distances,
                'positions': positions,
                'global_distances': global_distances,
                'visible_cells_per_position': visible_cells_per_position
            })

            # Add positions to global collection
            for i, ((x, y), visible_cells) in enumerate(zip(positions, visible_cells_per_position, strict=True)):
                all_position_data.append({
                    'position': (x, y),
                    'visible_cells': visible_cells,
                    'path_idx': path_idx,
                    'position_idx': i,
                    'global_distance': global_distances[i]
                })

            global_time_offset += path.length

        # Second pass: calculate metrics using the view model
        if all_position_data:
            # Calculate total likelihood by tracking all observed cells
            all_observed_cells = set()
            for position_data in all_position_data:
                all_observed_cells.update(position_data['visible_cells'])

            # Calculate total likelihood (sum of all unique observed cells)
            total_likelihood = sum(self.heatmap[row, col] for row, col in all_observed_cells)

            # Calculate time-discounted score considering all visible cells at each position
            total_time_discounted_score = 0
            for position_data in all_position_data:
                position_score = sum(self.heatmap[row, col] for row, col in position_data['visible_cells'])
                discount = discount_factor ** position_data['global_distance']
                total_time_discounted_score += position_score * discount
        else:
            total_likelihood = 0
            total_time_discounted_score = 0

        # Third pass: generate cumulative results for each path
        cumulative_distances_all_paths = []
        cumulative_likelihoods_all_paths = []
        cumulative_discounted_scores_all_paths = []

        for meta in path_metadata:
            if not meta['positions']:
                cumulative_distances_all_paths.append(np.array([0]))
                cumulative_likelihoods_all_paths.append(np.array([0]))
                cumulative_discounted_scores_all_paths.append(np.array([0]))
                continue

            # Track cells observed by this path (without double-counting within the same path)
            path_observed_cells = set()
            path_likelihoods = []
            path_discounted_likelihoods = []

            for i, visible_cells in enumerate(meta['visible_cells_per_position']):
                # Only count cells not yet observed in this path
                new_cells = visible_cells - path_observed_cells
                path_observed_cells.update(new_cells)

                # Calculate likelihood for new cells only
                position_likelihood = sum(self.heatmap[row, col] for row, col in new_cells)
                discount = discount_factor ** meta['distances'][i]

                path_likelihoods.append(position_likelihood)
                path_discounted_likelihoods.append(position_likelihood * discount)

            cumulative_distances_all_paths.append(meta['distances'])
            cumulative_likelihoods_all_paths.append(np.cumsum(path_likelihoods))
            cumulative_discounted_scores_all_paths.append(np.cumsum(path_discounted_likelihoods))

        # # --- Geospatial Metrics (handled separately for efficiency) ---
        victim_metrics = self._calculate_victims_found_score(paths)
        area_covered = self._calculate_area_covered(paths)
        total_path_length = self._calculate_total_path_length(paths)

        # 4. Assemble the final results dictionary
        return {
            'total_likelihood_score': total_likelihood,
            'total_time_discounted_score': total_time_discounted_score,
            'victim_detection_metrics': victim_metrics,
            'area_covered': area_covered,
            'total_path_length': total_path_length,
            'cumulative_distances': cumulative_distances_all_paths,
            'cumulative_likelihoods': cumulative_likelihoods_all_paths,
            'cumulative_time_discounted_scores': cumulative_discounted_scores_all_paths,
        }

    def _calculate_victims_found_score(self, paths: list[LineString]) -> dict:
        """
        Calculates victim detection percentage and timeliness.
        This is kept as a separate internal method as its logic is geospatial,
        not point-interpolation based.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths or self.victims.empty:
            return {'percentage_found': 0, 'found_victim_indices': []}

        # Memory-efficient approach: use unary_union to avoid intermediate geometry accumulation
        buffered_paths = [p.buffer(self.detection_radius) for p in valid_paths]
        coverage_area = unary_union(buffered_paths)

        # Clear the buffered_paths list to free memory
        del buffered_paths

        found_victims = self.victims[self.victims.within(coverage_area)]

        percentage_found = (len(found_victims) / len(self.victims)) * 100 if not self.victims.empty else 0

        return {
            'percentage_found': percentage_found,
            'found_victim_indices': found_victims.index.tolist()
        }

    def _calculate_area_covered(self, paths: list[LineString]) -> float:
        """
        Calculates the area covered by the paths within the detection radius.
        Handles overlapping paths by computing the union of all buffered areas.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.

        Returns:
            float: The total area covered by the paths in square kilometers, considering the detection radius,
                   with no double-counting of overlapping areas.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths:
            return 0.0

        # Memory-efficient approach: use unary_union to avoid intermediate geometry accumulation
        buffered_paths = [path.buffer(self.detection_radius) for path in valid_paths]
        combined_coverage = unary_union(buffered_paths)

        # Clear the buffered_paths list to free memory
        del buffered_paths

        return combined_coverage.area / 1_000_000  # Convert from m² to km²

    def _calculate_total_path_length(self, paths: list[LineString]) -> float:
        """
        Calculates the total length of all agent paths.

        Args:
            paths (list[LineString]): A list of Shapely LineString objects representing the drone paths.

        Returns:
            float: The total length of all paths in kilometers.
        """
        valid_paths = [p for p in paths if not p.is_empty and p.length > 0]
        if not valid_paths:
            return 0.0

        total_length = sum(path.length for path in valid_paths)
        return total_length / 1000  # Convert from meters to kilometers

    def get_visible_cells(self, x: float, y: float) -> set[tuple[int, int]]:
        """
        Get all grid cells that are visible from a given position based on the detection radius.
        
        Args:
            x (float): X coordinate in world space
            y (float): Y coordinate in world space
            
        Returns:
            set[tuple[int, int]]: Set of (row, col) tuples representing visible grid cells
        """
        minx, miny, maxx, maxy = self.extent
        
        # Convert detection radius to grid cells
        radius_in_cells_x = int(np.ceil(self.detection_radius / self.heatmap_cell_size_x))
        radius_in_cells_y = int(np.ceil(self.detection_radius / self.heatmap_cell_size_y))
        
        # Get the center cell position
        center_col = int((x - minx) / self.heatmap_cell_size_x)
        center_row = int((y - miny) / self.heatmap_cell_size_y)
        
        visible_cells = set()
        
        # Check all cells within the radius
        for row in range(max(0, center_row - radius_in_cells_y),
                        min(self.heatmap.shape[0], center_row + radius_in_cells_y + 1)):
            for col in range(max(0, center_col - radius_in_cells_x),
                            min(self.heatmap.shape[1], center_col + radius_in_cells_x + 1)):
                
                # Calculate the world coordinates of this cell's center
                cell_x = minx + (col + 0.5) * self.heatmap_cell_size_x
                cell_y = miny + (row + 0.5) * self.heatmap_cell_size_y
                
                # Check if this cell is within the detection radius
                distance = np.sqrt((cell_x - x) ** 2 + (cell_y - y) ** 2)
                if distance <= self.detection_radius:
                    visible_cells.add((row, col))
        
        return visible_cells

    def calculate_view_score_at_position(self, x: float, y: float, visited_cells: set[tuple[int, int]]) -> float:
        """
        Calculate the total likelihood score for all visible cells from a position,
        excluding already visited cells.
        
        Args:
            x (float): X coordinate in world space
            y (float): Y coordinate in world space
            visited_cells (set): Set of already visited (row, col) tuples
            
        Returns:
            float: Total likelihood score for unvisited visible cells
        """
        visible_cells = self.get_visible_cells(x, y)
        total_score = 0.0
        
        for row, col in visible_cells:
            if (row, col) not in visited_cells:
                total_score += self.heatmap[row, col]
        
        return total_score

    def calculate_metrics_at_distance_intervals(self, paths: list[LineString], discount_factor: float = 0.999, 
                                               interval_distance: float = 500.0) -> dict:
        """
        Efficiently calculate metrics at regular distance intervals along paths for animation purposes.
        This computes metrics every N meters along the path instead of every N positions.
        
        Args:
            paths (list[LineString]): List of drone paths
            discount_factor (float): Discount factor for time-based scoring
            interval_distance (float): Distance in meters between metric calculations
            
        Returns:
            dict: Contains precomputed metrics at regular distance intervals:
                - 'interval_metrics': List of metric dicts for each interval
                - 'interval_distances': List of cumulative distances for each interval
                - 'interval_positions': List of drone positions for each interval
                - 'total_intervals': Total number of intervals computed
        """
        # Flatten paths if they are nested (e.g., list of lists of LineStrings)
        flat_paths = []
        for item in paths:
            if isinstance(item, list):
                flat_paths.extend(item)
            else:
                flat_paths.append(item)
        
        valid_paths = [p for p in flat_paths if hasattr(p, 'is_empty') and not p.is_empty and p.length > 0]
        if not valid_paths:
            return {
                'interval_metrics': [],
                'interval_distances': [],
                'interval_positions': [],
                'total_intervals': 0
            }
        
        # Calculate maximum path length to determine total intervals needed
        max_path_length = max(path.length for path in valid_paths)
        total_intervals = int(np.ceil(max_path_length / interval_distance)) + 1
        
        interval_metrics = []
        interval_distances = []
        interval_position_list = []
        
        # Pre-compute path segments for efficiency based on distance
        path_segments = []
        path_coordinates = []  # Store all coordinates for natural path rendering
        for path_idx, path in enumerate(valid_paths):
            segments = []
            coordinates = []
            for i in range(total_intervals):
                distance_along_path = i * interval_distance
                
                if distance_along_path <= path.length:
                    # Get the position at this distance along the path
                    point = path.interpolate(distance_along_path)
                    point_coords = (point.x, point.y)
                    
                    # Create partial path up to this distance
                    if distance_along_path == 0:
                        # For the starting point, create a minimal path
                        start_point = path.interpolate(0)
                        partial_path = LineString([start_point.coords[0], start_point.coords[0]])
                        # Store just the starting point
                        path_coords_up_to_here = [start_point.coords[0]]
                    else:
                        # Create path from start to current distance with detailed points
                        num_points = max(10, int(distance_along_path / 50))  # Point every 50m for smooth rendering
                        distances = np.linspace(0, distance_along_path, num_points)
                        points = [path.interpolate(d) for d in distances]
                        partial_path = LineString([(p.x, p.y) for p in points])
                        # Store all coordinates up to this point for natural path rendering
                        path_coords_up_to_here = [(p.x, p.y) for p in points]
                else:
                    # Use full path if we've exceeded its length
                    partial_path = path
                    end_point = path.interpolate(path.length)
                    point_coords = (end_point.x, end_point.y)
                    distance_along_path = path.length
                    # Store all coordinates of the full path
                    path_coords_up_to_here = list(path.coords)
                
                segments.append({
                    'distance': distance_along_path,
                    'position': point_coords,
                    'partial_path': partial_path
                })
                coordinates.append(path_coords_up_to_here)
            path_segments.append(segments)
            path_coordinates.append(coordinates)
        
        # Calculate metrics for each distance interval
        for interval_idx in range(total_intervals):
            # Get partial paths up to this distance interval
            partial_paths = []
            positions = []
            path_coords_for_interval = []
            
            for drone_idx, (segments, coordinates) in enumerate(zip(path_segments, path_coordinates)):
                if interval_idx < len(segments):
                    partial_paths.append(segments[interval_idx]['partial_path'])
                    positions.append(segments[interval_idx]['position'])
                    path_coords_for_interval.append(coordinates[interval_idx])
                else:
                    # Use the last available segment (full path)
                    partial_paths.append(segments[-1]['partial_path'])
                    positions.append(segments[-1]['position'])
                    path_coords_for_interval.append(coordinates[-1])
            
            # Store path coordinates for this interval for natural rendering
            interval_position_list.append(positions)
            
            # Calculate metrics for this distance interval
            try:
                metrics_result = self.calculate_all_metrics(partial_paths, discount_factor)
                interval_metrics.append({
                    'area_covered': metrics_result.get('area_covered', 0),
                    'likelihood_score': metrics_result.get('total_likelihood_score', 0),
                    'victims_found_pct': metrics_result.get('victim_detection_metrics', {}).get('percentage_found', 0),
                    'total_path_length': metrics_result.get('total_path_length', 0),
                    'time_discounted_score': metrics_result.get('total_time_discounted_score', 0)
                })
            except Exception as e:
                # Use previous metrics or defaults if calculation fails
                if interval_metrics:
                    interval_metrics.append(interval_metrics[-1].copy())
                else:
                    interval_metrics.append({
                        'area_covered': 0, 'likelihood_score': 0, 'victims_found_pct': 0,
                        'total_path_length': 0, 'time_discounted_score': 0
                    })
            
            # Use the actual distance for this interval
            current_distance = interval_idx * interval_distance
            interval_distances.append(current_distance)
        
        return {
            'interval_metrics': interval_metrics,
            'interval_distances': interval_distances,
            'interval_positions': interval_position_list,
            'interval_path_coordinates': path_coordinates,  # All path coordinates for natural rendering
            'total_intervals': total_intervals,
            'interval_distance_step': interval_distance
        }

    def calculate_metrics_at_intervals(self, paths: list[LineString], discount_factor: float = 0.999, 
                                     interval_positions: int = 100) -> dict:
        """
        Efficiently calculate metrics at regular position intervals along paths for animation purposes.
        This is much faster than recalculating full metrics for every frame.
        
        Args:
            paths (list[LineString]): List of drone paths
            discount_factor (float): Discount factor for time-based scoring
            interval_positions (int): Number of positions between metric calculations
            
        Returns:
            dict: Contains precomputed metrics at regular intervals:
                - 'interval_metrics': List of metric dicts for each interval
                - 'interval_distances': List of cumulative distances for each interval
                - 'interval_positions': List of drone positions for each interval
                - 'total_intervals': Total number of intervals computed
        """
        # Flatten paths if they are nested (e.g., list of lists of LineStrings)
        flat_paths = []
        for item in paths:
            if isinstance(item, list):
                flat_paths.extend(item)
            else:
                flat_paths.append(item)
        
        valid_paths = [p for p in flat_paths if hasattr(p, 'is_empty') and not p.is_empty and p.length > 0]
        if not valid_paths:
            return {
                'interval_metrics': [],
                'interval_distances': [],
                'interval_positions': [],
                'total_intervals': 0
            }
        
        # Calculate maximum number of positions to determine total intervals needed
        max_positions = 0
        path_coords = []
        for path in valid_paths:
            coords = list(path.coords)
            path_coords.append(coords)
            if isinstance(coords, list) and len(coords) > 0:
                max_positions = max(max_positions, len(coords))
        
        total_intervals = int(np.ceil(max_positions / interval_positions)) + 1
        
        interval_metrics = []
        interval_distances = []
        interval_position_list = []  # Renamed to avoid collision with parameter
        
        # Pre-compute path segments for efficiency
        path_segments = []
        for path_idx, (path, coords) in enumerate(zip(valid_paths, path_coords)):
            segments = []
            for i in range(total_intervals):
                position_idx = min(i * interval_positions, len(coords) - 1)
                
                if position_idx < len(coords):
                    # Get the actual position from coordinates
                    point_coords = coords[position_idx]
                    
                    # Create partial path up to this position
                    if position_idx == 0:
                        partial_path = LineString([coords[0], coords[0]])
                    else:
                        # Use coordinates up to this position index
                        partial_coords = coords[:position_idx + 1]
                        partial_path = LineString(partial_coords) if len(partial_coords) > 1 else LineString([partial_coords[0], partial_coords[0]])
                    
                    # Calculate distance along path
                    if position_idx == 0:
                        distance = 0
                    else:
                        temp_path = LineString(coords[:position_idx + 1])
                        distance = temp_path.length
                else:
                    # Use full path if we've exceeded its length
                    partial_path = path
                    point_coords = coords[-1]
                    distance = path.length
                
                segments.append({
                    'distance': distance,
                    'position': point_coords,
                    'partial_path': partial_path,
                    'position_idx': position_idx
                })
            path_segments.append(segments)
        
        # Calculate metrics for each interval
        for interval_idx in range(total_intervals):
            # Get partial paths up to this interval
            partial_paths = []
            positions = []
            
            for drone_idx, segments in enumerate(path_segments):
                if interval_idx < len(segments):
                    partial_paths.append(segments[interval_idx]['partial_path'])
                    positions.append(segments[interval_idx]['position'])
                else:
                    # Use the last available segment
                    partial_paths.append(segments[-1]['partial_path'])
                    positions.append(segments[-1]['position'])
            
            # Calculate metrics for this interval
            try:
                metrics_result = self.calculate_all_metrics(partial_paths, discount_factor)
                interval_metrics.append({
                    'area_covered': metrics_result.get('area_covered', 0),
                    'likelihood_score': metrics_result.get('total_likelihood_score', 0),
                    'victims_found_pct': metrics_result.get('victim_detection_metrics', {}).get('percentage_found', 0),
                    'total_path_length': metrics_result.get('total_path_length', 0),
                    'time_discounted_score': metrics_result.get('total_time_discounted_score', 0)
                })
            except Exception as e:
                # Use previous metrics or defaults if calculation fails
                if interval_metrics:
                    interval_metrics.append(interval_metrics[-1].copy())
                else:
                    interval_metrics.append({
                        'area_covered': 0, 'likelihood_score': 0, 'victims_found_pct': 0,
                        'total_path_length': 0, 'time_discounted_score': 0
                    })
            
            # Use average distance across all drones for this interval
            avg_distance = sum(path_segments[i][min(interval_idx, len(path_segments[i])-1)]['distance'] 
                             for i in range(len(path_segments))) / len(path_segments)
            interval_distances.append(avg_distance)
            interval_position_list.append(positions)
        
        return {
            'interval_metrics': interval_metrics,
            'interval_distances': interval_distances,
            'interval_positions': interval_position_list,
            'total_intervals': total_intervals,
            'interval_positions_step': interval_positions
        }

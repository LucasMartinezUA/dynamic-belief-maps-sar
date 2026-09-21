# sarenv/analytics/paths.py
"""
Collection of coverage path generation algorithms for drones.
"""
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from shapely.geometry import LineString, Point
from shapely.ops import substring

def split_path_for_drones(path: LineString, num_drones: int) -> list[LineString]:
    if num_drones <= 1 or path.is_empty or path.length == 0:
        return [path]
    segments = []
    segment_length = path.length / num_drones
    for i in range(num_drones):
        segments.append(substring(path, i * segment_length, (i + 1) * segment_length))
    return segments

# Add **kwargs to accept and ignore unused arguments
def generate_spiral_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    loop_spacing = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    a = loop_spacing / (2 * np.pi)
    num_rotations = max_radius / loop_spacing
    theta_max = num_rotations * 2 * np.pi
    approx_path_length = 0.5 * a * (theta_max * np.sqrt(1 + theta_max**2) + np.log(theta_max + np.sqrt(1 + theta_max**2))) if theta_max > 0 else 0
    num_points = int(approx_path_length / path_point_spacing_m) if path_point_spacing_m > 0 else 2000
    theta = np.linspace(0, theta_max, max(2, num_points))
    radius = np.clip(a * theta, 0, max_radius)
    full_path = LineString(zip(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta), strict=True))
    paths = split_path_for_drones(full_path, num_drones)
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

# Add **kwargs here as well for consistency
def generate_concentric_circles_path(center_x: float, center_y: float, max_radius: float, fov_deg: float, altitude: float, overlap: float, num_drones: int, path_point_spacing_m: float, transition_distance_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    radius_increment = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    path_points, current_radius = [], radius_increment
    while current_radius <= max_radius:
        transition_angle_rad = transition_distance_m / current_radius if current_radius > 0 else np.radians(45)
        arc_length = current_radius * (2 * np.pi - transition_angle_rad)
        num_points_circle = max(2, int(arc_length / path_point_spacing_m))
        theta = np.linspace(0, 2 * np.pi - transition_angle_rad, num_points_circle)
        path_points.extend(zip(center_x + current_radius * np.cos(theta), center_y + current_radius * np.sin(theta), strict=True))
        next_radius = current_radius + radius_increment
        if next_radius <= max_radius:
            path_points.append((center_x + next_radius, center_y))
        else:
            final_theta_points = max(2, int((current_radius * transition_angle_rad) / path_point_spacing_m))
            final_theta = np.linspace(2 * np.pi - transition_angle_rad, 2 * np.pi, final_theta_points)
            path_points.extend(zip(center_x + current_radius * np.cos(final_theta), center_y + current_radius * np.sin(final_theta), strict=True))
        current_radius = next_radius
    full_path = LineString(path_points) if path_points else LineString()
    paths = split_path_for_drones(full_path, num_drones)
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

# Add **kwargs here too to handle arguments like 'transition_distance_m'
def generate_pizza_zigzag_path(center_x: float, center_y: float, max_radius: float, num_drones: int, fov_deg: float, altitude: float, overlap: float, path_point_spacing_m: float, border_gap_m: float, **kwargs) -> list[LineString]:
    budget = kwargs.get('budget')
    paths, section_angle_rad = [], 2 * np.pi / num_drones
    pass_width = (2 * altitude * np.tan(np.radians(fov_deg / 2))) * (1 - overlap)
    for i in range(num_drones):
        base_start_angle, base_end_angle = i * section_angle_rad, (i + 1) * section_angle_rad
        points, radius, direction = [(center_x, center_y)], pass_width, 1
        while radius <= max_radius:
            angular_offset_rad = border_gap_m / radius if radius > 0 else 0
            start_angle, end_angle = base_start_angle + angular_offset_rad, base_end_angle - angular_offset_rad
            if start_angle >= end_angle:
                radius += pass_width
                continue
            arc_length = radius * (end_angle - start_angle)
            num_arc_points = max(2, int(arc_length / path_point_spacing_m))
            current_arc_angles = np.linspace(start_angle, end_angle, num_arc_points) if direction == 1 else np.linspace(end_angle, start_angle, num_arc_points)
            points.extend(zip(center_x + radius * np.cos(current_arc_angles), center_y + radius * np.sin(current_arc_angles), strict=True))
            radius += pass_width
            direction *= -1
        if len(points) > 1:
            paths.append(LineString(points))
    
    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(paths, budget / num_drones)
    
    return paths

def generate_greedy_path(center_x: float, center_y: float, num_drones: int, probability_map: np.ndarray, bounds: tuple, max_radius: float, **kwargs) -> list[LineString]:
    """Generate greedy coverage paths for multiple drones.
    
    Supports N-step lookahead via convolution + maximum filter (Bellman equation).
    Set lookahead_steps=N in kwargs to enable N-step planning (N >= 2).
    Pass seed=int to get reproducible tie-breaking.
    """
    height, width = probability_map.shape
    minx, miny, maxx, maxy = bounds
    rng = np.random.default_rng(kwargs.get('seed', None))

    if maxx <= minx or maxy <= miny:
        return [LineString() for _ in range(num_drones)]

    # Extract view model parameters
    fov_deg = kwargs.get('fov_deg', 45.0)  # Default FOV
    altitude = kwargs.get('altitude', 80.0)  # Default altitude
    lookahead_steps = kwargs.get('lookahead_steps', 1)  # N-step lookahead
    detection_radius = altitude * np.tan(np.radians(fov_deg / 2))

    # Pre-compute coordinate mappings for efficiency
    dx = (maxx - minx) / width
    dy = ((maxy - miny) / height )
    x_offset = minx + dx / 2
    y_offset = miny + dy / 2

    # Convert detection radius to grid cells
    detection_radius_cells_x = int(np.ceil(detection_radius / dx))
    detection_radius_cells_y = int(np.ceil(detection_radius / dy))

    # Build circular FOV kernel for convolution-based lookahead
    if lookahead_steps >= 2:
        Y, X = np.ogrid[-detection_radius_cells_y:detection_radius_cells_y + 1,
                        -detection_radius_cells_x:detection_radius_cells_x + 1]
        dist_grid = np.sqrt((X * dx) ** 2 + (Y * dy) ** 2)
        fov_kernel = (dist_grid <= detection_radius).astype(np.float64)
    else:
        fov_kernel = None

    def get_visible_cells_from_grid_pos(row: int, col: int) -> set[tuple[int, int]]:
        """Get all grid cells visible from a grid position considering detection radius."""
        visible_cells = set()
        world_x = x_offset + col * dx
        world_y = y_offset + row * dy
        
        # Check all cells within the detection radius
        for r in range(max(0, row - detection_radius_cells_y),
                      min(height, row + detection_radius_cells_y + 1)):
            for c in range(max(0, col - detection_radius_cells_x),
                          min(width, col + detection_radius_cells_x + 1)):
                
                # Calculate world coordinates of this cell's center
                cell_x = x_offset + c * dx
                cell_y = y_offset + r * dy
                
                # Check if within detection radius
                distance = np.sqrt((cell_x - world_x) ** 2 + (cell_y - world_y) ** 2)
                if distance <= detection_radius:
                    visible_cells.add((r, c))
        
        return visible_cells

    def calculate_position_score(row: int, col: int, observed_cells: set) -> float:
        """Calculate the score for a position based on unobserved visible cells."""
        visible_cells = get_visible_cells_from_grid_pos(row, col)
        new_cells = visible_cells - observed_cells
        return sum(probability_map[r, c] for r, c in new_cells)

    # Check if initial_positions are provided
    initial_positions = kwargs.get('initial_positions')
    if initial_positions is not None and len(initial_positions) >= num_drones:
        # Convert world coordinates to grid positions
        current_positions = []
        for wx, wy in initial_positions[:num_drones]:
            col = np.clip(int((wx - minx) / dx), 0, width - 1)
            row = np.clip(int((wy - miny) / dy), 0, height - 1)
            current_positions.append((row, col))
    else:
        # Use default center-based initialization
        start_col = np.clip(int((center_x - minx) / dx), 0, width - 1)
        start_row = np.clip(int((center_y - miny) / dy), 0, height - 1)
        start_pos = (start_row, start_col)

        # Initialize drone positions efficiently
        current_positions = [start_pos]
        for i in range(1, num_drones):
            angle = 2 * np.pi * i / num_drones
            offset_r = min(2, height // 10)
            offset_c = min(2, width // 10)
            new_r = np.clip(start_pos[0] + int(offset_r * np.sin(angle)), 0, height - 1)
            new_c = np.clip(start_pos[1] + int(offset_c * np.cos(angle)), 0, width - 1)
            current_positions.append((new_r, new_c))

    # Pre-compute squared max radius for faster distance checks
    max_radius_sq = max_radius * max_radius

    # Track globally observed cells across all drones
    globally_observed_cells = set()
    # For N-step: also maintain a numpy mask for fast V recomputation
    observed_mask = np.zeros((height, width), dtype=np.float64) if lookahead_steps >= 2 else None

    # Store paths as lists of (row, col) indices
    paths = [[] for _ in range(num_drones)]
    for i, pos in enumerate(current_positions):
        paths[i].append(pos)
        # Add visible cells from initial positions to globally observed
        visible_cells = get_visible_cells_from_grid_pos(pos[0], pos[1])
        globally_observed_cells.update(visible_cells)
        if observed_mask is not None:
            for r, c in visible_cells:
                observed_mask[r, c] = 1.0

    # Pre-define neighbor offsets (8-connected)
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    max_iterations = height * width // num_drones
    
    # Estimate max_iterations based on budget (in meters) and minimum move length
    budget = kwargs.get('budget')
    if budget is not None and budget > 0:
        # Each move is at least dx meters (cell width)
        max_iterations = (budget // dx) // num_drones

    iteration = 0
    V = None
    
    while iteration < max_iterations:
        iteration += 1
        
        # For N-step: recompute value map from updated observed mask (~14ms per call)
        if lookahead_steps >= 2 and fov_kernel is not None:
            unobserved = probability_map * (1.0 - observed_mask)
            score_map = convolve(unobserved, fov_kernel, mode='constant', cval=0.0)
            V = score_map
            for _ in range(lookahead_steps - 1):
                V = score_map + maximum_filter(V, size=3, mode='constant', cval=0.0)
        
        for i in range(num_drones):
            current_r, current_c = current_positions[i]
            valid_neighbors = []

            # Check all 8 neighbors
            for dr, dc in neighbor_offsets:
                nr, nc = current_r + dr, current_c + dc
                # Bounds check
                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue
                
                # Distance check using squared distance
                world_x = x_offset + nc * dx
                world_y = y_offset + nr * dy
                dist_sq = (world_x - center_x) ** 2 + (world_y - center_y) ** 2
                if dist_sq >= max_radius_sq:
                    continue

                if V is not None:
                    # N-step: use value map (recomputed with observed mask)
                    valid_neighbors.append(((nr, nc), V[nr, nc]))
                else:
                    # 1-step: calculate score based on unobserved cells
                    score = calculate_position_score(nr, nc, globally_observed_cells)
                    valid_neighbors.append(((nr, nc), score))

            # Choose next position based on scores
            if valid_neighbors:
                # Select the neighbor with the maximum score
                best_neighbor, max_prob = max(valid_neighbors, key=lambda x: x[1])
                
                if max_prob <= 0:
                    # Randomly choose from all valid neighbors when max score is 0 or below
                    best_neighbor = rng.choice(len(valid_neighbors))
                    best_neighbor = valid_neighbors[best_neighbor][0]
                current_positions[i] = best_neighbor
                
                paths[i].append(best_neighbor)
                
                # Update globally observed cells with newly visible cells
                visible_cells = get_visible_cells_from_grid_pos(best_neighbor[0], best_neighbor[1])
                globally_observed_cells.update(visible_cells)
                if observed_mask is not None:
                    for r, c in visible_cells:
                        observed_mask[r, c] = 1.0

    # Convert grid paths back to real-world coordinate paths
    line_paths = []
    for drone_path_indices in paths:
        if len(drone_path_indices) > 1:
            # Convert (row, col) indices directly to world coordinates
            line_paths.append(LineString([(x_offset + c * dx, y_offset + r * dy) for r, c in drone_path_indices]))
        else:
            line_paths.append(LineString())

    # Apply budget constraint if specified - trim excess points
    if budget is not None and budget > 0:
        paths = restrict_path_length(line_paths, budget / num_drones)
    else:
        paths = line_paths
    
    return paths

def generate_random_walk_path(
    center_x: float,
    center_y: float,
    num_drones: int,
    probability_map,
    **kwargs
) -> list[LineString]:
    return generate_greedy_path(
        center_x=center_x,
        center_y=center_y,
        num_drones=num_drones,
        probability_map=np.zeros_like(probability_map),
        **kwargs
    )

def restrict_path_length(line: LineString, max_length: float) -> LineString:
    if isinstance(line, list):
        return [restrict_path_length(path, max_length) for path in line]
    if line.is_empty or max_length is None or max_length <= 0 or line.length <= max_length:
        return line
    return substring(line, 0, max_length)

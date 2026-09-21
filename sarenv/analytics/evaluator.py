# toolkit.py
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import sarenv
from sarenv.analytics import metrics, paths
from sarenv.utils import geo
from sarenv.utils.logging_setup import get_logger
from sarenv.utils.plot import plot_single_evaluation_results

log = get_logger()

class PathGeneratorConfig:
    """
    Configuration class for path generation parameters.
    Provides a clean interface for managing path generation parameters.
    """

    def __init__(self, num_drones: int, budget: float, **kwargs):
        """
        Initialize path generation configuration.

        Args:
            num_drones (int): Number of drones to simulate. Required.
            budget (float): Budget constraint for path generation. Required.
            **kwargs: Additional keyword arguments for path generation parameters.
                fov_degrees (float): Field of view in degrees. Defaults to 45.0.
                altitude_meters (float): Altitude in meters. Defaults to 80.0.
                overlap_ratio (float): Overlap ratio for systematic patterns. Defaults to 0.25.
                path_point_spacing_m (float): Spacing between path points in meters. Defaults to 10.0.
                transition_distance_m (float): Transition distance for concentric patterns. Defaults to 50.0.
                pizza_border_gap_m (float): Border gap for pizza patterns. Defaults to 15.0.
        """
        self.num_drones = num_drones
        self.budget = budget
        self.fov_degrees = kwargs.pop('fov_degrees', 45.0)
        self.altitude_meters = kwargs.pop('altitude_meters', 80.0)
        self.overlap_ratio = kwargs.pop('overlap_ratio', 0)
        self.path_point_spacing_m = kwargs.pop('path_point_spacing_m', 10.0)
        self.transition_distance_m = kwargs.pop('transition_distance_m', 50.0)
        self.pizza_border_gap_m = kwargs.pop('pizza_border_gap_m', 15.0)

        # Store any additional parameters not explicitly defined
        self.additional_params = kwargs

    def get_params_dict(
        self,
        center_x: float,
        center_y: float,
        max_radius: float,
        probability_map: np.ndarray | None,
        bounds: tuple[float, float, float, float] | None,
    ) -> dict[str, float | int | np.ndarray | tuple | None]:
        """
        Generate a complete parameter dictionary for path generation.

        Args:
            center_x: X coordinate of the center point
            center_y: Y coordinate of the center point
            max_radius: Maximum search radius
            probability_map: Optional probability map for informed search
            bounds: Optional geographic bounds tuple

        Returns:
            Dictionary with all parameters needed for path generation
        """
        params = {
            'center_x': center_x,
            'center_y': center_y,
            'max_radius': max_radius,
            'probability_map': probability_map,
            'bounds': bounds,
            'num_drones': self.num_drones,
            'budget': self.budget,
            'fov_deg': self.fov_degrees,
            'altitude': self.altitude_meters,
            'overlap': self.overlap_ratio,
            'path_point_spacing_m': self.path_point_spacing_m,
            'transition_distance_m': self.transition_distance_m,
            'border_gap_m': self.pizza_border_gap_m,
        }

        # Add any additional parameters
        params.update(self.additional_params)

        return params

class PathGenerator:
    """
    Wrapper class for path generation functions that provides a consistent interface.
    Uses PathGeneratorConfig to ensure consistent parameter handling.
    """

    def __init__(self, name: str, func, path_generator_config: PathGeneratorConfig, description: str = ""):
        """
        Initialize a path generator.

        Args:
            name: Name of the generator
            func: Function that generates paths
            description: Optional description of the generator
        """
        self.name = name
        self.func = func
        self.description = description
        self.path_generator_config = path_generator_config

    def __call__(
        self,
        center_x: float,
        center_y: float,
        max_radius: float,
        probability_map: np.ndarray | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> list[LineString]:
        """
        Call the generator function with properly configured parameters.

        Args:
            center_x: X coordinate of the center point
            center_y: Y coordinate of the center point
            max_radius: Maximum radius for the search area
            probability_map: Optional probability map for informed search
            bounds: Optional geographic bounds tuple

        Returns:
            List of LineString objects representing paths for each drone
        """
        # Get all parameters from the config
        params = self.path_generator_config.get_params_dict(
            center_x=center_x,
            center_y=center_y,
            max_radius=max_radius,
            probability_map=probability_map,
            bounds=bounds,
        )

        # All parameters must be passed to ensure consistent behavior
        return self.func(**params)


def get_default_path_generators(config: PathGeneratorConfig) -> dict[str, PathGenerator]:
    """
    Get default path generators with consistent parameter handling.

    Args:
        config: Configuration object containing default parameters for path generation

    Returns:
        Dictionary of PathGenerator instances
    """
    return {
        "RandomWalk": PathGenerator(
            name="RandomWalk",
            func=paths.generate_random_walk_path,
            path_generator_config=config,
            description="Random walk path generation"
        ),
        "Greedy": PathGenerator(
            name="Greedy",
            func=paths.generate_greedy_path,
            path_generator_config=config,
            description="Greedy path generation based on probability map"
        ),
        "Spiral": PathGenerator(
            name="Spiral",
            func=paths.generate_spiral_path,
            path_generator_config=config,
            description="Spiral path generation"
        ),
        "Concentric": PathGenerator(
            name="Concentric",
            func=paths.generate_concentric_circles_path,
            path_generator_config=config,
            description="Concentric circles path generation"
        ),
        "Pizza": PathGenerator(
            name="Pizza",
            func=paths.generate_pizza_zigzag_path,
            path_generator_config=config,
            description="Pizza slice zigzag path generation"

        )
    }


class ComparativeDatasetEvaluator:
    """
    Evaluates multiple datasets using ComparativeEvaluator to minimize code duplication.
    Combines all results into DataFrames and saves them to CSV files for further analysis.

    This class focuses solely on running experiments and managing data - all plotting
    and visualization should be handled externally using the returned DataFrames.
    """

    def __init__(self,
                 dataset_dirs,
                 evaluation_sizes,
                 num_drones,
                 budget,
                 num_lost_persons=100,
                 **kwargs):
        """
        Initializes the ComparativeDatasetEvaluator.

        Args:
            dataset_dirs (list): A list of directories for the datasets to be evaluated.
                                Defaults to ["sarenv_dataset/1", "sarenv_dataset/2", "sarenv_dataset/3", "sarenv_dataset/4"].
            evaluation_sizes (list): List of dataset sizes to evaluate. Defaults to ["medium"].
            num_drones (int): Number of drones to simulate. Defaults to 1.
            num_lost_persons (int): Number of victim locations to generate. Defaults to 100.
            budget (int): Budget constraint for path generation. Defaults to 100,000.
            **kwargs: Additional keyword arguments for evaluation and path generation.
                path_generator_config (PathGeneratorConfig): Configuration for path parameters.
                path_generators (dict): Dict of {name: PathGenerator} instances.
                discount_factor (float): Discount factor for time-based scores. Defaults to 0.999.
                fov_degrees (float): Camera field of view in degrees. Defaults to 45.0.
                altitude_meters (float): Drone altitude in meters. Defaults to 80.0.
                overlap_ratio (float): Overlap ratio for patterns. Defaults to 0.25.
                path_point_spacing_m (float): Spacing between path points. Defaults to 10.0.
                transition_distance_m (float): Transition distance for concentric patterns. Defaults to 50.0.
                pizza_border_gap_m (float): Border gap for pizza patterns. Defaults to 15.0.
        """
        # Set default parameters
        self.dataset_dirs = dataset_dirs or [f"sarenv_dataset/{i}" for i in range(1, 5)]
        self.evaluation_sizes = evaluation_sizes or ["medium"]
        self.num_drones = num_drones
        self.num_victims = num_lost_persons
        self.budget = budget
        self.discount_factor = kwargs.get("discount_factor", 0.999)

        # Extract path generator configuration
        path_generators = kwargs.get("path_generators")
        path_generator_config = kwargs.get("path_generator_config")

        # Create path generator configuration if not provided
        if path_generator_config is None:
            self.path_generator_config = PathGeneratorConfig(num_drones=self.num_drones, budget=self.budget,**kwargs.copy())
        else:
            self.path_generator_config = path_generator_config

        # Set up path generators
        if path_generators is None:
            self.path_generators = get_default_path_generators(self.path_generator_config)
        else:
            self.path_generators = {}
            for name, generator in path_generators.items():
                if isinstance(generator, PathGenerator):
                    self.path_generators[name] = generator
                else:
                    # Assume it's a function and wrap it
                    self.path_generators[name] = PathGenerator(
                        name=name,
                        func=generator,
                        path_generator_config=self.path_generator_config,
                        description=f"Custom generator: {name}"
                    )


        # Storage for combined results
        self.metrics_results = []
        self.time_series_results = []
        self.path_results = []  # Store generated paths

    def evaluate(self, output_dir):
        """
        Evaluates all path generators across all datasets using ComparativeEvaluator instances.
        Saves results to CSV files immediately after each dataset evaluation to optimize memory usage.

        Args:
            output_dir (str): Directory to save the results files. Defaults to "results".

        Returns:
            tuple: (metrics_df, time_series_df) - Combined DataFrames with all results
        """
        # Create output directory if it doesn't exist
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Prepare file name suffix with number of drones and budget
            suffix = f"_n{self.path_generator_config.num_drones}_budget{self.path_generator_config.budget}"
            metrics_file = Path(output_dir) / f"comparative_metrics_results{suffix}.csv"
            time_series_file = Path(output_dir) / f"comparative_time_series_results{suffix}.csv"
            
            # Initialize CSV files with headers
            metrics_header_written = False
            time_series_header_written = False
        
        # Create a ComparativeEvaluator for each dataset
        self.evaluators = []

        for dataset_dir in self.dataset_dirs:
            # Pass necessary parameters to ComparativeEvaluator
            evaluator = ComparativeEvaluator(
                dataset_directory=dataset_dir,
                evaluation_sizes=self.evaluation_sizes,
                num_lost_persons=self.num_victims,
                budget=self.budget,  # Pass the budget from ComparativeDatasetEvaluator
                path_generator_config=self.path_generator_config,
                path_generators=self.path_generators,
                # Pass num_drones from the config explicitly
                num_drones=self.path_generator_config.num_drones
            )
            self.evaluators.append(evaluator)

            log.info(f"Evaluating dataset: {dataset_dir}")

            # Run evaluation for this dataset
            results_df, time_series_data = evaluator.run_baseline_evaluations()

            if results_df.empty:
                continue

            # Process metrics results and save immediately
            dataset_metrics_results = []
            for _, row in results_df.iterrows():
                # Convert the entire row to a dictionary and add dataset directory name
                result_dict = row.to_dict()
                result_dict["Dataset"] = Path(dataset_dir).name
                dataset_metrics_results.append(result_dict)

            # Process time-series data and save immediately
            dataset_time_series_results = []
            for algorithm_name, time_series_list in time_series_data.items():
                for i, time_series_data in enumerate(time_series_list):
                    # Process individual drone data
                    individual_drone_data = time_series_data.get('individual_drone_data', [])
                    
                    # Get the environment size from the results for this algorithm
                    algorithm_results = results_df[results_df['Algorithm'] == algorithm_name]
                    environment_size = algorithm_results['Environment Size'].iloc[0] if not algorithm_results.empty else "unknown"

                    # Create rows for each drone's timesteps
                    for drone_data in individual_drone_data:
                        drone_id = drone_data['drone_id']
                        cumulative_likelihood = drone_data['cumulative_likelihood']
                        positions = drone_data['positions']

                        # Calculate per-drone victims (distribute total across drones)
                        total_victims_found = results_df.iloc[-1]["Victims Found (%)"] / 100.0 * self.num_victims if not results_df.empty else 0
                        per_drone_victims = total_victims_found / len(individual_drone_data) if individual_drone_data else 0

                        for t, likelihood in enumerate(cumulative_likelihood):
                            # Get the path position for this timestep
                            path_x, path_y = None, None
                            if t < len(positions):
                                path_x, path_y = positions[t]

                            # Calculate cumulative victims for this drone
                            drone_victims = per_drone_victims * (t + 1) / len(cumulative_likelihood) if len(cumulative_likelihood) > 0 else 0

                            dataset_time_series_results.append({
                                "Algorithm": algorithm_name,
                                "Dataset": Path(dataset_dir).name,
                                "Environment Size": environment_size,
                                "Run": i,
                                "Agent_ID": drone_id,
                                "Time_Step": t,
                                "Cumulative_Likelihood": likelihood,
                                "Cumulative_Victims": drone_victims,
                                "Path_X": path_x,
                                "Path_Y": path_y,
                            })

            # Save results immediately to CSV files
            if output_dir is not None and dataset_metrics_results:
                dataset_metrics_df = pd.DataFrame(dataset_metrics_results)
                dataset_time_series_df = pd.DataFrame(dataset_time_series_results)
                
                # Save metrics results (append to existing file)
                dataset_metrics_df.to_csv(metrics_file, mode='a', header=not metrics_header_written, index=False)
                metrics_header_written = True
                
                # Save time-series results (append to existing file)
                if not dataset_time_series_df.empty:
                    dataset_time_series_df.to_csv(time_series_file, mode='a', header=not time_series_header_written, index=False)
                    time_series_header_written = True
                
                log.info(f"Saved results for dataset {Path(dataset_dir).name} to CSV files")
                
                # Keep minimal results in memory for return values
                self.metrics_results.extend(dataset_metrics_results)
                self.time_series_results.extend(dataset_time_series_results)
                
                # Clear the dataset-specific results to free memory
                del dataset_metrics_results
                del dataset_time_series_results
                del dataset_metrics_df
                del dataset_time_series_df
                
                # Periodically clear memory to prevent accumulation
                self._clear_memory()
            else:
                # If no output_dir specified, keep in memory
                self.metrics_results.extend(dataset_metrics_results)
                self.time_series_results.extend(dataset_time_series_results)

        # Create final DataFrames from the collected results (minimal memory usage)
        metrics_df = pd.DataFrame(self.metrics_results)
        time_series_df = pd.DataFrame(self.time_series_results)

        if output_dir is not None:
            log.info("All results saved to:")
            log.info(f"  Metrics: {metrics_file}")
            log.info(f"  Time series: {time_series_file}")

        return metrics_df, time_series_df

    def save_results(self, metrics_df: pd.DataFrame, time_series_df: pd.DataFrame, output_dir: str = "results"):
        """
        Save the metrics, time-series, and paths results to CSV files.
        Note: This method is kept for backward compatibility, but evaluation now saves results incrementally.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing metrics results
            time_series_df (pd.DataFrame): DataFrame containing time-series results
            output_dir (str): Directory to save the files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Prepare file name suffix with number of drones and budget
        suffix = f"_n{self.path_generator_config.num_drones}_budget{self.path_generator_config.budget}"

        # Save metrics results
        metrics_file = Path(output_dir) / f"comparative_metrics_results{suffix}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        log.info(f"Saved metrics results to: {metrics_file}")

        # Save time-series results
        time_series_file = Path(output_dir) / f"comparative_time_series_results{suffix}.csv"
        time_series_df.to_csv(time_series_file, index=False)
        log.info(f"Saved time-series results to: {time_series_file}")

        log.info("Note: Results are now saved incrementally during evaluation for better memory efficiency.")

    def get_metrics_results(self) -> pd.DataFrame:
        """
        Returns the metrics results as a DataFrame.
        """
        return pd.DataFrame(self.metrics_results)

    def get_time_series_results(self) -> pd.DataFrame:
        """
        Returns the time-series results as a DataFrame.
        """
        return pd.DataFrame(self.time_series_results)

    def get_paths_results(self) -> pd.DataFrame:
        """
        Returns the paths results as a DataFrame.
        """
        return pd.DataFrame(self.path_results)

    def get_results_per_dataset(self) -> pd.DataFrame:
        """
        Returns the results as a DataFrame grouped by dataset.
        """
        return pd.DataFrame(self.metrics_results).groupby("Dataset").apply(lambda x: x.reset_index(drop=True))

    def summarize_results(self) -> pd.DataFrame:
        """
        Returns a summary DataFrame grouped by algorithm with means and 95% confidence intervals.
        """
        results_df = pd.DataFrame(self.metrics_results)
        if results_df.empty:
            return pd.DataFrame()

        return results_df.groupby("Algorithm").agg(
            Mean_Likelihood_Score=('Likelihood Score', 'mean'),
            CI_Likelihood_Score=('Likelihood Score', lambda x: 1.96 * x.sem()),
            Mean_Time_Discounted=('Time-Discounted Score', 'mean'),
            CI_Time_Discounted=('Time-Discounted Score', lambda x: 1.96 * x.sem()),
            Mean_Victims_Found=('Victims Found (%)', 'mean'),
            CI_Victims_Found=('Victims Found (%)', lambda x: 1.96 * x.sem()),
            Mean_Area_Covered=('Area Covered (km²)', 'mean'),
            CI_Area_Covered=('Area Covered (km²)', lambda x: 1.96 * x.sem()),
            Mean_Path_Length=('Total Path Length (km)', 'mean'),
            CI_Path_Length=('Total Path Length (km)', lambda x: 1.96 * x.sem()),
        ).reset_index()

    def _clear_memory(self):
        """
        Clear stored results from memory to optimize memory usage.
        This is called internally during evaluation to free up memory.
        """
        # Clear the in-memory results but keep a small subset for return values
        if len(self.metrics_results) > 1000:  # Keep only last 1000 results
            self.metrics_results = self.metrics_results[-1000:]
        if len(self.time_series_results) > 10000:  # Keep only last 10000 results
            self.time_series_results = self.time_series_results[-10000:]


class ComparativeEvaluator:
    """
    A toolkit for running and comparing UAV pathfinding algorithms.

    This class simplifies the process of evaluating multiple pathfinding
    strategies across different datasets and visualizing the results.
    """

    def __init__(self,
                 dataset_directory="sarenv_dataset",
                 evaluation_sizes=None,
                 num_drones=1,
                 num_lost_persons=100,
                 budget=100_000,
                 seed=42,
                 **kwargs):
        """
        Initializes the ComparativeEvaluator.

        Args:
            dataset_directory (str): Path to the sarenv dataset. Defaults to "sarenv_dataset".
            evaluation_sizes (list): List of dataset sizes to evaluate. Defaults to ["small", "medium", "large"].
            num_drones (int): The number of drones to simulate. Defaults to 1.
            num_lost_persons (int): Number of victim locations to generate. Defaults to 100.
            budget (int): Budget constraint for path generation. Defaults to 100,000.
            **kwargs: Additional keyword arguments for evaluation.
                path_generator_config (PathGeneratorConfig): Configuration for path parameters.
                path_generators (dict): Dict of {name: PathGenerator} instances.
                use_defaults (bool): Whether to use default path generators if none are provided. Defaults to True.
        """
        self.dataset_directory = dataset_directory
        self.evaluation_sizes = evaluation_sizes or ["small", "medium", "large"]
        self.num_victims = num_lost_persons
        self.num_drones = num_drones
        self.budget = budget
        self.seed = seed
        # Create path generator configuration with explicit parameters
        self.path_generator_config = PathGeneratorConfig(num_drones=self.num_drones, budget=self.budget,**kwargs.copy())
        self.path_generators = kwargs.get("path_generators")

        self.loader = sarenv.DatasetLoader(dataset_directory=self.dataset_directory)
        self.environments = {}
        self.results = None
        self.time_series_data = {}

        # Set up path generators, defaulting if none are provided
        if self.path_generators is None:
            self.path_generators = get_default_path_generators(self.path_generator_config)
        else:
            # Wrap custom generators with PathGenerator if needed
            wrapped_generators = {}
            for name, generator in self.path_generators.items():
                if isinstance(generator, PathGenerator):
                    wrapped_generators[name] = generator
                else:
                    # Assume it's a function and wrap it
                    wrapped_generators[name] = PathGenerator(
                        name=name,
                        func=generator,
                        path_generator_config=self.path_generator_config,
                        description=f"Custom generator: {name}"
                    )
            self.path_generators = wrapped_generators

        self.load_datasets()


    def load_datasets(self):
        """
        Loads all specified datasets and generates static victim locations.
        This prepares the evaluator for running the simulations.
        """
        log.info(f"Loading datasets for sizes: {self.evaluation_sizes}")
        for size in self.evaluation_sizes:
            item = self.loader.load_environment(size)

            if not item:
                log.warning(f"Could not load data for size '{size}'. Skipping.")
                continue

            data_crs = geo.get_utm_epsg(item.center_point[0], item.center_point[1])
            victim_generator = sarenv.LostPersonLocationGenerator(item, seed=self.seed)
            victim_points = [
                p
                for p in (
                    victim_generator.generate_location()
                    for _ in range(self.num_victims)
                )
                if p
            ]
            victims_gdf = (
                gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
                if victim_points
                else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
            )

            self.environments[size] = {
                "item": item,
                "victims": victims_gdf,
                "crs": data_crs,
            }
        log.info("All datasets loaded and prepared.")

    def run_baseline_evaluations(self) -> tuple[pd.DataFrame, dict]:
        """
        Runs all baseline algorithms across all loaded datasets.

        Returns:
            tuple: (metrics_df, time_series_data) - A tuple containing the metrics DataFrame
                   and time series data dictionary.
        """
        if not self.environments:
            log.error("No datasets loaded. Please call 'load_datasets()' first.")
            return pd.DataFrame()

        all_results = []
        self.time_series_data = {}  # Reset time-series data for each evaluation

        for size, env_data in self.environments.items():
            item = env_data["item"]
            victims_gdf = env_data["victims"]

            log.info(f"--- Evaluating Baselines on '{size}' dataset ---")

            evaluator = metrics.PathEvaluator(
                item.heatmap,
                item.bounds,
                victims_gdf,
                self.path_generator_config.fov_degrees,
                self.path_generator_config.altitude_meters,
                self.loader._meter_per_bin
            )
            center_proj = (
                gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
                .to_crs(env_data["crs"])
                .geometry.iloc[0]
            )

            for name, generator in self.path_generators.items():
                log.info(f"Running {name} algorithm on '{size}' dataset...")
                generator : PathGenerator
                generated_paths = generator(
                    center_proj.x,
                    center_proj.y,
                    item.radius_km * 1000,
                    item.heatmap,
                    item.bounds,
                )

                all_metrics = evaluator.calculate_all_metrics(
                    generated_paths,
                    0.999
                )

                # Extract results from the returned dictionary
                victim_metrics = all_metrics['victim_detection_metrics']

                result = {
                    "Dataset": size,
                    "Algorithm": name,
                    "Environment Type": item.environment_type,
                    "Climate": item.environment_climate,
                    "Environment Size": size,
                    "n_agents": self.path_generator_config.num_drones,
                    "Budget (m)": self.path_generator_config.budget,
                    "Likelihood Score": all_metrics['total_likelihood_score'],
                    "Time-Discounted Score": all_metrics['total_time_discounted_score'],
                    "Victims Found (%)": victim_metrics['percentage_found'],
                    "Area Covered (km²)": all_metrics['area_covered'],
                    "Total Path Length (km)": all_metrics['total_path_length'],
                }

                all_results.append(result)

                # Collect time-series data for this algorithm
                if name not in self.time_series_data:
                    self.time_series_data[name] = []

                # Calculate combined cumulative metrics for time series
                cumulative_likelihoods = all_metrics['cumulative_likelihoods']
                if cumulative_likelihoods:
                    # Instead of combining, we'll store individual drone data with agent IDs
                    individual_drone_data = []

                    for drone_idx, cum_lik in enumerate(cumulative_likelihoods):
                        if len(cum_lik) > 0 and drone_idx < len(generated_paths):
                            drone_path = generated_paths[drone_idx]

                            # Get path positions for this specific drone
                            drone_positions = []
                            if not drone_path.is_empty and drone_path.length > 0:
                                # Use the same interpolation logic as in PathEvaluator
                                interpolation_resolution = int(np.ceil(self.loader._meter_per_bin / 2))
                                num_points = int(np.ceil(drone_path.length / interpolation_resolution)) + 1
                                distances = np.linspace(0, drone_path.length, num_points)

                                # Interpolate points along the path
                                for d in distances:
                                    point = drone_path.interpolate(d)
                                    drone_positions.append((point.x, point.y))

                                # If we have more timesteps than path positions, pad with the last position
                                while len(drone_positions) < len(cum_lik):
                                    if drone_positions:
                                        drone_positions.append(drone_positions[-1])
                                    else:
                                        drone_positions.append((0, 0))

                                # If we have more path positions than timesteps, truncate
                                drone_positions = drone_positions[:len(cum_lik)]
                            else:
                                # Empty path case
                                drone_positions = [(0, 0)] * len(cum_lik)

                            # Store individual drone data
                            individual_drone_data.append({
                                'drone_id': drone_idx,
                                'cumulative_likelihood': cum_lik,
                                'positions': drone_positions
                            })
                else:
                    individual_drone_data = []

                # Store time-series results with only individual drone data
                self.time_series_data[name].append({
                    'individual_drone_data': individual_drone_data,
                })

        self.results = pd.DataFrame(all_results)
        log.info("--- Baseline Evaluation Complete ---")
        log.info(f"Results:\n{self.results.to_string()}")
        return self.results, self.time_series_data

    def plot_results(self, results_df: pd.DataFrame = None, output_dir="graphs"):
        """
        Generates and saves plots for the evaluation results.

        Args:
            results_df (pd.DataFrame, optional): The dataframe of results to plot.
                                                 If None, uses the last stored results.
            output_dir (str): The directory to save the plots in.
        """
        if results_df is None:
            results_df = self.results

        plot_single_evaluation_results(results_df, self.evaluation_sizes, output_dir)

"""Command-line interface for the inverse ephys workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .biological_data import (
    EXPANDED_COMMON_FEATURES,
    INTRINSIC_COMMON_FEATURES,
    SPIKE_CYCLE_COMMON_FEATURES,
    WAVEFORM_CORE_FEATURES,
    download_biological_data,
    harmonize_model_features,
    load_biological_features,
    load_local_spike_cycle_features,
)
from .calibration import calibrate_parameter_bounds, expanded_kinetic_bounds
from .coverage import (
    analyze_coverage,
    compare_feature_distributions,
    save_coverage_tables,
    save_distribution_tables,
)
from .dataset import DatasetConfig, evaluate_parameter_frame, generate_dataset
from .features import FeatureConfig
from .hh_model import SimulationConfig, Stimulus, simulate
from .prediction import save_predictor, train_predictor
from .protocols import (
    SCREEN_DEFAULT_TEMPERATURE_C,
    SCREEN_PRESETS,
    biological_screen_config,
)
from .raw_patchseq import discover_local_nwb, extract_local_spike_cycles
from .sampling import KINETIC_SAMPLING_MODES


def _generate(args: argparse.Namespace) -> None:
    if args.kinetics_only and args.static_only:
        raise ValueError("--kinetics-only and --static-only are mutually exclusive")
    output = Path(args.output)
    rejected_output = output.with_name(f"{output.stem}_rejected.csv")
    metadata_output = output.with_name(f"{output.stem}_metadata.json")
    if args.temperature is not None:
        temperature_c = args.temperature
    elif args.protocol == "biological-screen":
        temperature_c = SCREEN_DEFAULT_TEMPERATURE_C[args.screen_preset]
    else:
        temperature_c = 6.3
    stimulus = Stimulus(
        amplitude_ua_cm2=args.current if args.current_pa is None else None,
        amplitude_pa=args.current_pa,
        start_ms=args.stim_start,
        end_ms=args.stim_end,
    )
    simulation_config = SimulationConfig(
        duration_ms=args.duration,
        dt_ms=args.dt,
        stimulus=stimulus,
        temperature_c=temperature_c,
    )
    parameter_bounds = None
    if args.bounds is not None:
        bounds_document = json.loads(Path(args.bounds).read_text(encoding="utf-8"))
        parameter_bounds = bounds_document.get("bounds", bounds_document)
    dataset_config = DatasetConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        workers=args.workers,
        include_efel=args.efel,
        include_static=not args.kinetics_only,
        sample_kinetics=not args.static_only,
        kinetic_sampling_mode=args.kinetic_sampling,
        protocol=args.protocol,
        parameter_bounds=parameter_bounds,
    )
    feature_config = FeatureConfig(min_spikes=args.min_spikes)
    screen_config = None
    if args.protocol == "biological-screen":
        screen_config = biological_screen_config(
            args.screen_preset,
            temperature_c=temperature_c,
            dt_ms=args.dt,
        )
        screen_config = replace(
            screen_config,
            rheobase_tolerance_pa=args.rheobase_tolerance_pa,
            waveform_current_offset_pa=args.waveform_offset_pa,
        )
    accepted, rejected, metadata = generate_dataset(
        dataset_config,
        simulation_config,
        feature_config,
        screen_config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output, index=False)
    rejected.to_csv(rejected_output, index=False)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Accepted {len(accepted)}/{args.n_samples} models "
        f"({metadata['acceptance_fraction']:.1%})."
    )
    print(f"Dataset: {output}")
    print(f"Rejected samples: {rejected_output}")
    print(f"Metadata: {metadata_output}")


def _train(args: argparse.Namespace) -> None:
    from .plots import save_recovery_plot

    dataset = pd.read_csv(args.dataset)
    bundle, metrics, predictions = train_predictor(
        dataset,
        seed=args.seed,
        test_fraction=args.test_fraction,
        n_estimators=args.n_estimators,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictor(bundle, output_dir / "parameter_predictor.joblib")
    metrics.to_csv(output_dir / "parameter_metrics.csv", index=False)
    predictions.to_csv(output_dir / "held_out_predictions.csv", index=False)
    bundle["feature_importance"].to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    save_recovery_plot(
        predictions,
        metrics,
        output_dir / "parameter_recovery.png",
    )
    print(f"Mean held-out R2: {metrics['r2'].mean():.3f}")
    print(f"Results: {output_dir}")


def _rescreen(args: argparse.Namespace) -> None:
    parameters = pd.read_csv(args.parameters)
    temperature_c = (
        SCREEN_DEFAULT_TEMPERATURE_C[args.screen_preset]
        if args.temperature is None
        else args.temperature
    )
    screen_config = biological_screen_config(
        args.screen_preset,
        temperature_c=temperature_c,
        dt_ms=args.dt,
    )
    screen_config = replace(
        screen_config,
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
        waveform_current_offset_pa=args.waveform_offset_pa,
    )
    dataset_config = DatasetConfig(
        n_samples=len(parameters),
        workers=args.workers,
        protocol="biological-screen",
    )
    accepted, rejected, metadata = evaluate_parameter_frame(
        parameters,
        dataset_config=dataset_config,
        screen_config=screen_config,
    )
    output = Path(args.output)
    rejected_output = output.with_name(f"{output.stem}_rejected.csv")
    metadata_output = output.with_name(f"{output.stem}_metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(output, index=False)
    rejected.to_csv(rejected_output, index=False)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Accepted {len(accepted)}/{len(parameters)} rescreened models "
        f"({metadata['acceptance_fraction']:.1%})."
    )
    print(f"Dataset: {output}")
    print(f"Rejected samples: {rejected_output}")
    print(f"Metadata: {metadata_output}")


def _inspect(args: argparse.Namespace) -> None:
    from .plots import save_trace_plot

    trace = simulate()
    save_trace_plot(trace, args.output)
    print(f"Canonical HH diagnostic: {args.output}")


def _fetch_biological(args: argparse.Namespace) -> None:
    paths = download_biological_data(args.data_dir, force=args.force)
    print(f"Downloaded or verified {len(paths)} processed source files.")
    print(f"Biological data: {args.data_dir}")


def _extract_local_nwb(args: argparse.Namespace) -> None:
    if args.scala_root is None and args.gouwens_root is None:
        raise ValueError("Provide --scala-root, --gouwens-root, or both")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    inventory = discover_local_nwb(
        scala_root=args.scala_root,
        gouwens_root=args.gouwens_root,
    )
    if inventory.empty:
        raise ValueError("No NWB files found under the supplied roots")
    if args.limit_per_dataset is not None:
        inventory = inventory.groupby("dataset", sort=False, dropna=False).head(
            args.limit_per_dataset
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    if args.resume and output.exists():
        previous = pd.read_csv(output, dtype={"cell_id": "string"})
        frames.append(previous)
        completed = set(
            zip(previous["dataset"].astype(str), previous["cell_id"].astype(str))
        )
        inventory = inventory.loc[
            [
                (str(dataset), str(cell_id)) not in completed
                for dataset, cell_id in zip(
                    inventory["dataset"],
                    inventory["cell_id"],
                )
            ]
        ].copy()

    total_remaining = len(inventory)
    for start in range(0, total_remaining, args.checkpoint_every):
        chunk = inventory.iloc[start : start + args.checkpoint_every]
        frames.append(
            extract_local_spike_cycles(
                chunk,
                workers=args.workers,
            )
        )
        extracted = pd.concat(frames, ignore_index=True, sort=False)
        temporary = output.with_suffix(f"{output.suffix}.part")
        extracted.to_csv(temporary, index=False)
        temporary.replace(output)
        print(
            f"Processed {min(start + len(chunk), total_remaining)}/"
            f"{total_remaining} remaining files.",
            flush=True,
        )
    if not frames:
        extracted = pd.DataFrame()
    else:
        extracted = pd.concat(frames, ignore_index=True, sort=False)
    counts = extracted.groupby(["dataset", "raw_status"], dropna=False).size()
    print(counts.to_string())
    print(f"Raw spike-cycle features: {output}")


def _coverage(args: argparse.Namespace) -> None:
    from .plots import (
        save_coverage_plot,
        save_feature_distribution_violin_plot,
    )

    if args.profile == "spike-cycle":
        if args.raw_spike_cycles is None:
            raise ValueError("--profile spike-cycle requires --raw-spike-cycles")
        biological = load_local_spike_cycle_features(
            args.raw_spike_cycles,
            args.data_dir,
            include_scala_physiological_temperature=not args.exclude_scala_phys_temp,
        )
    else:
        biological = load_biological_features(
            args.data_dir,
            include_scala_physiological_temperature=not args.exclude_scala_phys_temp,
        )
    if args.biological_dataset != "all":
        biological = biological.loc[
            biological["dataset"] == args.biological_dataset
        ].copy()
        if biological.empty:
            raise ValueError(
                f"No biological rows found for dataset {args.biological_dataset}"
            )
    model_dataset = pd.read_csv(args.models)
    models = harmonize_model_features(model_dataset)
    feature_profiles = {
        "waveform-core": WAVEFORM_CORE_FEATURES,
        "expanded-common": EXPANDED_COMMON_FEATURES,
        "intrinsic": INTRINSIC_COMMON_FEATURES,
        "spike-cycle": SPIKE_CYCLE_COMMON_FEATURES,
    }
    features = feature_profiles[args.profile]
    result = analyze_coverage(
        biological,
        models,
        features,
        local_neighbor_rank=args.local_neighbor_rank,
    )
    output_dir = Path(args.output_dir)
    save_coverage_tables(result, output_dir)
    distribution_result = compare_feature_distributions(
        biological,
        models,
        features,
    )
    save_distribution_tables(distribution_result, output_dir)
    biological.to_csv(output_dir / "harmonized_biological_features.csv", index=False)
    models.to_csv(output_dir / "harmonized_model_features.csv", index=False)
    save_coverage_plot(result, output_dir / "feature_space_coverage.png")
    dataset_labels = {
        "all": "Pooled biological cohorts",
        "gouwens_visp": "Gouwens VISp / 34 C",
        "scala_room_temperature": "Scala room temperature / 22 C",
        "scala_physiological_temperature": "Scala physiological temperature",
    }
    profile_label = args.profile.replace("-", " ")
    save_feature_distribution_violin_plot(
        distribution_result,
        output_dir / "feature_distribution_violins.png",
        title=f"{dataset_labels[args.biological_dataset]}: {profile_label} features",
    )
    total = result.summary.loc[
        result.summary["dataset"] == "all_biological"
    ].iloc[0]
    print(
        f"Covered {total['coverage_fraction']:.1%} of "
        f"{int(total['n_biological_cells'])} complete biological cells "
        f"with {int(total['n_models'])} HH models."
    )
    print(f"Coverage results: {output_dir}")


def _direct_optimize(args: argparse.Namespace) -> None:
    from .direct_optimization import (
        DirectOptimizationConfig,
        optimize_biological_targets,
        save_direct_optimization,
    )

    biological = load_local_spike_cycle_features(
        args.raw_spike_cycles,
        args.data_dir,
        include_scala_physiological_temperature=False,
    )
    biological = biological.loc[
        biological["dataset"].eq(args.biological_dataset)
    ].copy()
    if biological.empty:
        raise ValueError(
            f"No biological rows found for dataset {args.biological_dataset}"
        )
    models = pd.read_csv(args.models)
    config = DirectOptimizationConfig(
        n_targets=args.n_targets,
        rounds=args.rounds,
        batch_size=args.batch_size,
        elite_count=args.elite_count,
        validation_count=args.validation_count,
        long_promotions_per_round=args.long_promotions_per_round,
        workers=args.workers,
        seed=args.seed,
    )
    result = optimize_biological_targets(
        biological,
        models,
        bounds_path=args.bounds,
        temperature_c=args.temperature,
        long_screen_preset=args.long_screen_preset,
        config=config,
    )
    save_direct_optimization(result, args.output_dir)
    print(result.target_summary.to_string(index=False))
    print(f"Direct optimization results: {args.output_dir}")


def _cell_optimize(args: argparse.Namespace) -> None:
    from .cell_optimization import (
        CellOptimizationConfig,
        optimize_cell,
        save_pareto_objective_plot,
        save_cell_optimization,
        save_representative_trace_plot,
    )
    from .cell_targets import (
        build_cell_optimization_target,
        load_raw_spike_cycle_table,
    )

    raw_features = load_raw_spike_cycle_table(args.raw_spike_cycles)
    target = build_cell_optimization_target(
        raw_features,
        dataset=args.biological_dataset,
        cell_id=args.cell_id,
        temperature_c=args.temperature,
    )
    seeds = (
        pd.read_csv(args.seed_models)
        if args.seed_models is not None
        else None
    )
    screen_config = biological_screen_config(
        args.screen_preset,
        temperature_c=args.temperature,
        dt_ms=args.dt,
    )
    screen_config = replace(
        screen_config,
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
    )
    config = CellOptimizationConfig(
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        seed=args.seed,
        gna_gk_factor=args.gna_gk_factor,
    )
    result = optimize_cell(
        target,
        bounds_path=args.bounds,
        screen_config=screen_config,
        config=config,
        seed_population=seeds,
        checkpoint_dir=args.output_dir,
    )
    save_cell_optimization(result, args.output_dir)
    save_pareto_objective_plot(
        result,
        Path(args.output_dir) / "pareto_objectives.png",
    )
    save_representative_trace_plot(
        result,
        screen_config,
        config,
        Path(args.output_dir) / "representative_trace_comparison.png",
    )
    objectives = [
        column
        for column in result.representative.columns
        if column in result.metadata["objective_names"]
    ]
    print(
        result.representative[
            ["representative_score", *objectives]
        ].to_string(index=False)
    )
    print(f"Held-out scores:\n{result.validation.to_string(index=False)}")
    print(f"Cell optimization results: {args.output_dir}")


def _ladder_optimize(args: argparse.Namespace) -> None:
    from .cell_optimization import CELL_PARAMETER_NAMES
    from .cell_targets import (
        build_cell_optimization_target,
        load_raw_spike_cycle_table,
    )
    from .model_ladder import (
        LadderOptimizationConfig,
        baseline_ladder_result,
        ladder_summary,
        optimize_ladder_variant,
        save_ladder_fit,
        save_ladder_summary_plot,
    )

    raw_features = load_raw_spike_cycle_table(args.raw_spike_cycles)
    target = build_cell_optimization_target(
        raw_features,
        dataset=args.biological_dataset,
        cell_id=args.cell_id,
        temperature_c=args.temperature,
    )
    base_table = pd.read_csv(args.base_model)
    if len(base_table) != 1:
        raise ValueError("--base-model must contain exactly one fitted model")
    missing = set(CELL_PARAMETER_NAMES).difference(base_table.columns)
    if missing:
        raise ValueError(f"Base model is missing parameters: {sorted(missing)}")
    base_vector = base_table.loc[
        base_table.index[0],
        list(CELL_PARAMETER_NAMES),
    ].to_numpy(dtype=float)
    screen_config = replace(
        biological_screen_config(
            args.screen_preset,
            temperature_c=args.temperature,
            dt_ms=args.dt,
        ),
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
    )
    config = LadderOptimizationConfig(
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        seed=args.seed,
        sigma=args.sigma,
    )
    output = Path(args.output_dir)
    results = [
        baseline_ladder_result(
            base_vector,
            target,
            screen_config,
            config,
        )
    ]
    save_ladder_fit(results[0], output / "base", screen_config)
    for variant in args.variants:
        print(f"Optimizing {variant}...", flush=True)
        result = optimize_ladder_variant(
            variant,
            base_vector,
            target,
            screen_config,
            config,
            checkpoint_dir=output / variant,
        )
        save_ladder_fit(result, output / variant, screen_config)
        results.append(result)
    summary = ladder_summary(results)
    summary.to_csv(output / "ladder_summary.csv", index=False)
    save_ladder_summary_plot(summary, output / "ladder_summary.png")
    print(
        summary[
            [
                "variant",
                "scalar_score",
                "firing_pattern",
                "held_out__firing_pattern",
                "model_rheobase_pa",
            ]
        ].to_string(index=False)
    )
    print(f"Model ladder results: {output}")


def _staged_ladder_optimize(args: argparse.Namespace) -> None:
    from .cell_targets import (
        build_cell_optimization_target,
        load_raw_spike_cycle_table,
    )
    from .staged_ladder import (
        StagedLadderConfig,
        optimize_staged_ladder,
        save_staged_result,
        staged_summary,
    )

    raw_features = load_raw_spike_cycle_table(args.raw_spike_cycles)
    target = build_cell_optimization_target(
        raw_features,
        dataset=args.biological_dataset,
        cell_id=args.cell_id,
        temperature_c=args.temperature,
        local_passive=True,
    )
    seeds = pd.read_csv(args.seed_models)
    screen_config = replace(
        biological_screen_config(
            args.screen_preset,
            temperature_c=args.temperature,
            dt_ms=args.dt,
        ),
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
    )
    config = StagedLadderConfig(
        fast_population_size=args.fast_population_size,
        fast_generations=args.fast_generations,
        fast_elites=args.fast_elites,
        slow_population_size=args.slow_population_size,
        slow_generations=args.slow_generations,
        final_population_size=args.final_population_size,
        final_generations=args.final_generations,
        workers=args.workers,
        seed=args.seed,
        trust_fraction=args.trust_fraction,
        voltage_shift_margin_mv=args.voltage_shift_margin_mv,
        gna_gk_factor=args.gna_gk_factor,
    )
    result = optimize_staged_ladder(
        target,
        bounds_path=args.bounds,
        seed_population=seeds,
        screen_config=screen_config,
        config=config,
        checkpoint_dir=args.output_dir,
    )
    save_staged_result(result, args.output_dir, screen_config)
    print(
        staged_summary(result)[
            [
                "stage",
                "scalar_score",
                "passive",
                "spike_shape",
                "phase_geometry",
                "firing_pattern",
                "held_out__firing_pattern",
            ]
        ].to_string(index=False)
    )
    print(f"Staged ladder results: {args.output_dir}")


def _phase_shape_experiment(args: argparse.Namespace) -> None:
    from .cell_optimization import CELL_PARAMETER_NAMES
    from .cell_targets import (
        build_cell_optimization_target,
        load_raw_spike_cycle_table,
    )
    from .phase_experiment import (
        PhaseExperimentConfig,
        phase_experiment_summary,
        run_phase_adequacy_experiment,
        save_phase_adequacy_result,
    )
    from .phase_shape import PhaseShapeConfig

    raw_features = load_raw_spike_cycle_table(args.raw_spike_cycles)
    target = build_cell_optimization_target(
        raw_features,
        dataset=args.biological_dataset,
        cell_id=args.cell_id,
        temperature_c=args.temperature,
        local_passive=True,
    )
    parent_table = pd.read_csv(args.parent_models)
    if args.n_parents < 1:
        raise ValueError("--n-parents must be at least one")
    missing = set(CELL_PARAMETER_NAMES).difference(parent_table.columns)
    if missing:
        raise ValueError(
            f"Parent models are missing parameters: {sorted(missing)}"
        )
    parents = parent_table.iloc[
        : args.n_parents,
    ].loc[
        :,
        list(CELL_PARAMETER_NAMES),
    ].to_numpy(dtype=float)
    screen_config = replace(
        biological_screen_config(
            args.screen_preset,
            temperature_c=args.temperature,
            dt_ms=args.dt,
        ),
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
    )
    experiment_config = PhaseExperimentConfig(
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        seed=args.seed,
        sigma=args.sigma,
        phase_step_duration_ms=args.phase_step_duration_ms,
        gna_gk_factor=args.gna_gk_factor,
    )
    shape_config = PhaseShapeConfig(
        grid_points=args.phase_grid_points,
        spline_smoothing=args.spline_smoothing,
    )
    result = run_phase_adequacy_experiment(
        target,
        parents,
        args.bounds,
        screen_config,
        experiment_config,
        shape_config,
    )
    save_phase_adequacy_result(result, args.output_dir)
    print(
        phase_experiment_summary(result)[
            [
                "architecture",
                "scalar_score",
                "total",
                "concavity",
                "onset_concavity_sign_agreement",
                "velocity_extent",
                "kinetics__pathology_score",
            ]
        ].to_string(index=False)
    )
    print(f"Phase-shape experiment results: {args.output_dir}")


def _phase_pareto_experiment(args: argparse.Namespace) -> None:
    from .cell_targets import (
        build_cell_optimization_target,
        load_raw_spike_cycle_table,
    )
    from .phase_pareto import (
        PhaseParetoConfig,
        nested_phase_summary,
        run_nested_phase_pareto_experiment,
        save_nested_phase_pareto_result,
    )
    from .phase_shape import PhaseShapeConfig

    raw_features = load_raw_spike_cycle_table(args.raw_spike_cycles)
    target = build_cell_optimization_target(
        raw_features,
        dataset=args.biological_dataset,
        cell_id=args.cell_id,
        temperature_c=args.temperature,
        local_passive=True,
    )
    baseline = pd.read_csv(args.baseline_model)
    screen_config = replace(
        biological_screen_config(
            args.screen_preset,
            temperature_c=args.temperature,
            dt_ms=args.dt,
        ),
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
    )
    experiment_config = PhaseParetoConfig(
        population_size=args.population_size,
        generations=args.generations,
        workers=args.workers,
        seed=args.seed,
        initialization_fraction=args.initialization_fraction,
        phase_step_duration_ms=args.phase_step_duration_ms,
        gna_gk_factor=args.gna_gk_factor,
    )
    shape_config = PhaseShapeConfig(
        grid_points=args.phase_grid_points,
        spline_smoothing=args.spline_smoothing,
    )
    result = run_nested_phase_pareto_experiment(
        target,
        baseline,
        args.bounds,
        screen_config,
        experiment_config,
        shape_config,
    )
    save_nested_phase_pareto_result(result, args.output_dir)
    print(
        nested_phase_summary(result)[
            [
                "variant",
                "parameter_count",
                "pareto_models",
                "joint_target_models",
                "total",
                "concavity",
                "velocity_extent",
                "onset_concavity_sign_agreement",
            ]
        ].to_string(index=False)
    )
    print(f"Nested phase Pareto results: {args.output_dir}")


def _direct_phase_fit(args: argparse.Namespace) -> None:
    from .cell_targets import (
        build_cell_optimization_target,
        load_raw_spike_cycle_table,
    )
    from .direct_phase_fit import (
        DirectPhaseFitConfig,
        direct_phase_summary,
        optimize_direct_phase_fit,
        save_direct_phase_fit_result,
    )
    from .phase_shape import PhaseShapeConfig

    raw_features = load_raw_spike_cycle_table(args.raw_spike_cycles)
    target = build_cell_optimization_target(
        raw_features,
        dataset=args.biological_dataset,
        cell_id=args.cell_id,
        temperature_c=args.temperature,
        local_passive=True,
    )
    baseline = pd.read_csv(args.baseline_model)
    screen_config = replace(
        biological_screen_config(
            args.screen_preset,
            temperature_c=args.temperature,
            dt_ms=args.dt,
        ),
        rheobase_tolerance_pa=args.rheobase_tolerance_pa,
    )
    fit_config = DirectPhaseFitConfig(
        shape_population_size=args.shape_population_size,
        shape_generations=args.shape_generations,
        shape_elites=args.shape_elites,
        scale_population_size=args.scale_population_size,
        scale_generations=args.scale_generations,
        final_population_size=args.final_population_size,
        final_generations=args.final_generations,
        workers=args.workers,
        seed=args.seed,
        sigma=args.sigma,
        gna_gk_factor=args.gna_gk_factor,
        phase_step_duration_ms=args.phase_step_duration_ms,
        trust_fraction=args.trust_fraction,
    )
    shape_config = PhaseShapeConfig(
        grid_points=args.phase_grid_points,
        spline_smoothing=args.spline_smoothing,
        voltage_extremum_tolerance_mv=args.voltage_tolerance_mv,
        velocity_constraint_fraction=args.velocity_tolerance_fraction,
    )
    result = optimize_direct_phase_fit(
        target,
        baseline,
        screen_config,
        fit_config,
        shape_config,
    )
    save_direct_phase_fit_result(
        result,
        screen_config,
        args.output_dir,
    )
    print(
        direct_phase_summary(result)[
            [
                "stage",
                "objective_total",
                "shape_loss",
                "scale_loss",
                "physical_constraint_loss",
                "rheobase_constraint_loss",
                "downstroke_low_voltage_concavity",
                "constraints_satisfied",
                "held_out__total",
            ]
        ].to_string(index=False)
    )
    print(f"Direct x_inf/tau phase-fit results: {args.output_dir}")


def _calibrate_prior(args: argparse.Namespace) -> None:
    models = pd.read_csv(args.models)
    plausibility = pd.read_csv(args.model_plausibility)
    result = calibrate_parameter_bounds(
        models,
        plausibility,
        parameter_group=args.parameter_group,
        minimum_models=args.minimum_models,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table_output = output.with_name(f"{output.stem}_summary.csv")
    selected_output = output.with_name(f"{output.stem}_selected_models.csv")
    bounds = dict(result.bounds)
    if args.kinetic_expansion > 1.0:
        bounds.update(expanded_kinetic_bounds(args.kinetic_expansion))
    document = {
        "version": 1,
        "parameter_group": args.parameter_group,
        "kinetic_expansion": args.kinetic_expansion,
        "models": str(args.models),
        "model_plausibility": str(args.model_plausibility),
        "bounds": bounds,
    }
    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    result.parameter_summary.to_csv(table_output, index=False)
    result.selected_models.to_csv(selected_output, index=False)
    print(
        f"Wrote {len(bounds)} parameter bounds "
        f"from {len(result.selected_models)} nearest models."
    )
    print(f"Prior bounds: {output}")
    print(f"Parameter summary: {table_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer HH alpha/beta kinetics from electrophysiological features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Plot the canonical HH model")
    inspect_parser.add_argument(
        "--output", default="outputs/canonical_hh.png", help="Output PNG path"
    )
    inspect_parser.set_defaults(func=_inspect)

    fetch_parser = subparsers.add_parser(
        "fetch-biological",
        help="Download processed Scala and Gouwens Patch-seq features",
    )
    fetch_parser.add_argument("--data-dir", default="data/external")
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.set_defaults(func=_fetch_biological)

    raw_parser = subparsers.add_parser(
        "extract-local-nwb",
        help="Extract sampled-rheobase spike-cycle features from local Patch-seq NWB files",
    )
    raw_parser.add_argument(
        "--scala-root",
        default=None,
        help="Scala_pseq_data directory containing 000008",
    )
    raw_parser.add_argument(
        "--gouwens-root",
        default=None,
        help="Patch-seq AIBS directory containing ephys/ephys_files",
    )
    raw_parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Process only the first N files from each dataset",
    )
    raw_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of NWB files to process concurrently",
    )
    raw_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Atomically update the output after this many files",
    )
    raw_parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed dataset/cell pairs already present in the output",
    )
    raw_parser.add_argument(
        "--output",
        default="outputs/local_patchseq_spike_cycles.csv",
    )
    raw_parser.set_defaults(func=_extract_local_nwb)

    generate_parser = subparsers.add_parser(
        "generate", help="Sample kinetics and retain spiking models"
    )
    generate_parser.add_argument("--n-samples", type=int, default=256)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--workers", type=int, default=1)
    generate_parser.add_argument("--current", type=float, default=10.0)
    generate_parser.add_argument(
        "--current-pa",
        type=float,
        default=None,
        help="Use a total injected current in pA instead of current density",
    )
    generate_parser.add_argument(
        "--protocol",
        choices=("fixed", "biological-screen"),
        default="fixed",
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Recording temperature in Celsius (defaults depend on screen preset)",
    )
    generate_parser.add_argument(
        "--screen-preset",
        choices=SCREEN_PRESETS,
        default="scala",
        help="Long-step durations for the biological screening protocol",
    )
    generate_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
        help="Stop bisection when the firing boundary bracket is this narrow",
    )
    generate_parser.add_argument(
        "--waveform-offset-pa",
        type=float,
        default=0.0,
        help="Extract waveform features this many pA above refined rheobase",
    )
    generate_parser.add_argument("--duration", type=float, default=100.0)
    generate_parser.add_argument("--dt", type=float, default=0.025)
    generate_parser.add_argument("--stim-start", type=float, default=10.0)
    generate_parser.add_argument("--stim-end", type=float, default=90.0)
    generate_parser.add_argument("--min-spikes", type=int, default=2)
    generate_parser.add_argument("--efel", action="store_true")
    generate_parser.add_argument(
        "--kinetics-only",
        action="store_true",
        help="Keep canonical static membrane parameters",
    )
    generate_parser.add_argument(
        "--static-only",
        action="store_true",
        help="Keep canonical alpha/beta kinetics and sample static membrane parameters",
    )
    generate_parser.add_argument(
        "--bounds",
        default=None,
        help="JSON file containing per-parameter [lower, upper] bounds",
    )
    generate_parser.add_argument(
        "--kinetic-sampling",
        choices=KINETIC_SAMPLING_MODES,
        default="independent",
        help=(
            "Sample alpha/beta coordinates independently, as paired gate moves, "
            "or split the draw evenly between both strategies"
        ),
    )
    generate_parser.add_argument(
        "--output", default="outputs/hh_alpha_beta_dataset.csv"
    )
    generate_parser.set_defaults(func=_generate)

    rescreen_parser = subparsers.add_parser(
        "rescreen",
        help="Evaluate an existing parameter table with a long-step protocol",
    )
    rescreen_parser.add_argument("parameters", help="CSV containing sample_id and parameters")
    rescreen_parser.add_argument("--workers", type=int, default=1)
    rescreen_parser.add_argument("--temperature", type=float, default=None)
    rescreen_parser.add_argument(
        "--screen-preset",
        choices=SCREEN_PRESETS,
        default="scala",
    )
    rescreen_parser.add_argument("--dt", type=float, default=0.025)
    rescreen_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    rescreen_parser.add_argument(
        "--waveform-offset-pa",
        type=float,
        default=0.0,
    )
    rescreen_parser.add_argument(
        "--output",
        default="outputs/rescreened_models.csv",
    )
    rescreen_parser.set_defaults(func=_rescreen)

    train_parser = subparsers.add_parser(
        "train", help="Predict kinetic parameters from accepted-model features"
    )
    train_parser.add_argument("dataset")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--test-fraction", type=float, default=0.25)
    train_parser.add_argument("--n-estimators", type=int, default=400)
    train_parser.add_argument("--output-dir", default="outputs/prediction")
    train_parser.set_defaults(func=_train)

    coverage_parser = subparsers.add_parser(
        "coverage",
        help="Compare accepted HH models with biological Patch-seq feature space",
    )
    coverage_parser.add_argument("models", help="Accepted HH model dataset CSV")
    coverage_parser.add_argument("--data-dir", default="data/external")
    coverage_parser.add_argument(
        "--profile",
        choices=("waveform-core", "expanded-common", "intrinsic", "spike-cycle"),
        default="waveform-core",
    )
    coverage_parser.add_argument(
        "--raw-spike-cycles",
        default=None,
        help="CSV generated by extract-local-nwb; required for spike-cycle coverage",
    )
    coverage_parser.add_argument("--local-neighbor-rank", type=int, default=5)
    coverage_parser.add_argument(
        "--biological-dataset",
        choices=(
            "all",
            "gouwens_visp",
            "scala_room_temperature",
            "scala_physiological_temperature",
        ),
        default="all",
        help="Fit coverage geometry to one protocol cohort or all cohorts",
    )
    coverage_parser.add_argument("--exclude-scala-phys-temp", action="store_true")
    coverage_parser.add_argument("--output-dir", default="outputs/coverage")
    coverage_parser.set_defaults(func=_coverage)

    direct_parser = subparsers.add_parser(
        "direct-optimize",
        help="Fit HH parameters directly to representative biological spike targets",
    )
    direct_parser.add_argument("models", help="Accepted HH model population CSV")
    direct_parser.add_argument("--bounds", required=True)
    direct_parser.add_argument("--raw-spike-cycles", required=True)
    direct_parser.add_argument("--data-dir", default="data/external")
    direct_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    direct_parser.add_argument("--temperature", type=float, required=True)
    direct_parser.add_argument(
        "--long-screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    direct_parser.add_argument("--n-targets", type=int, default=2)
    direct_parser.add_argument("--rounds", type=int, default=3)
    direct_parser.add_argument("--batch-size", type=int, default=32)
    direct_parser.add_argument("--elite-count", type=int, default=12)
    direct_parser.add_argument("--validation-count", type=int, default=3)
    direct_parser.add_argument(
        "--long-promotions-per-round",
        type=int,
        default=4,
        help="Candidates per round evaluated with the matched long protocol",
    )
    direct_parser.add_argument("--workers", type=int, default=1)
    direct_parser.add_argument("--seed", type=int, default=42)
    direct_parser.add_argument(
        "--output-dir",
        default="outputs/direct_optimization",
    )
    direct_parser.set_defaults(func=_direct_optimize)

    cell_parser = subparsers.add_parser(
        "cell-optimize",
        help="Fit alpha/beta kinetics to passive and multi-sweep data from one cell",
    )
    cell_parser.add_argument(
        "raw_spike_cycles",
        help="CSV generated by extract-local-nwb",
    )
    cell_parser.add_argument("--bounds", required=True)
    cell_parser.add_argument(
        "--seed-models",
        default=None,
        help="Optional accepted-model CSV used to seed half the population",
    )
    cell_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    cell_parser.add_argument("--cell-id", required=True)
    cell_parser.add_argument("--temperature", type=float, required=True)
    cell_parser.add_argument(
        "--screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    cell_parser.add_argument("--population-size", type=int, default=32)
    cell_parser.add_argument("--generations", type=int, default=10)
    cell_parser.add_argument("--workers", type=int, default=1)
    cell_parser.add_argument("--seed", type=int, default=42)
    cell_parser.add_argument("--dt", type=float, default=0.025)
    cell_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    cell_parser.add_argument(
        "--gna-gk-factor",
        type=float,
        default=1.5,
        help="Allow each maximal conductance within this factor of classic HH",
    )
    cell_parser.add_argument(
        "--output-dir",
        default="outputs/cell_optimization",
    )
    cell_parser.set_defaults(func=_cell_optimize)

    ladder_parser = subparsers.add_parser(
        "ladder-optimize",
        help="Compare minimal slow-gate and soma/AIS extensions with CMA-ES",
    )
    ladder_parser.add_argument(
        "raw_spike_cycles",
        help="CSV generated by extract-local-nwb",
    )
    ladder_parser.add_argument(
        "--base-model",
        required=True,
        help="One-row representative_model.csv from cell-optimize",
    )
    ladder_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    ladder_parser.add_argument("--cell-id", required=True)
    ladder_parser.add_argument("--temperature", type=float, required=True)
    ladder_parser.add_argument(
        "--screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    ladder_parser.add_argument(
        "--variants",
        nargs="+",
        choices=("na-slow", "k-slow", "soma-ais"),
        default=("na-slow", "k-slow", "soma-ais"),
    )
    ladder_parser.add_argument("--population-size", type=int, default=8)
    ladder_parser.add_argument("--generations", type=int, default=4)
    ladder_parser.add_argument("--workers", type=int, default=1)
    ladder_parser.add_argument("--seed", type=int, default=42)
    ladder_parser.add_argument("--sigma", type=float, default=0.55)
    ladder_parser.add_argument("--dt", type=float, default=0.025)
    ladder_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    ladder_parser.add_argument(
        "--output-dir",
        default="outputs/model_ladder",
    )
    ladder_parser.set_defaults(func=_ladder_optimize)

    staged_parser = subparsers.add_parser(
        "staged-ladder-optimize",
        help="Fit local passive, diverse fast elites, slow K, and a joint trust region",
    )
    staged_parser.add_argument(
        "raw_spike_cycles",
        help="CSV generated by extract-local-nwb",
    )
    staged_parser.add_argument("--bounds", required=True)
    staged_parser.add_argument("--seed-models", required=True)
    staged_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    staged_parser.add_argument("--cell-id", required=True)
    staged_parser.add_argument("--temperature", type=float, required=True)
    staged_parser.add_argument(
        "--screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    staged_parser.add_argument("--fast-population-size", type=int, default=12)
    staged_parser.add_argument("--fast-generations", type=int, default=3)
    staged_parser.add_argument("--fast-elites", type=int, default=2)
    staged_parser.add_argument("--slow-population-size", type=int, default=6)
    staged_parser.add_argument("--slow-generations", type=int, default=2)
    staged_parser.add_argument("--final-population-size", type=int, default=8)
    staged_parser.add_argument("--final-generations", type=int, default=2)
    staged_parser.add_argument("--workers", type=int, default=1)
    staged_parser.add_argument("--seed", type=int, default=42)
    staged_parser.add_argument("--dt", type=float, default=0.025)
    staged_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    staged_parser.add_argument("--trust-fraction", type=float, default=0.20)
    staged_parser.add_argument(
        "--voltage-shift-margin-mv",
        type=float,
        default=4.0,
    )
    staged_parser.add_argument("--gna-gk-factor", type=float, default=1.5)
    staged_parser.add_argument(
        "--output-dir",
        default="outputs/staged_ladder",
    )
    staged_parser.set_defaults(func=_staged_ladder_optimize)

    phase_parser = subparsers.add_parser(
        "phase-shape-experiment",
        help="Compare one-compartment and soma-AIS fits to smoothed phase branches",
    )
    phase_parser.add_argument(
        "raw_spike_cycles",
        help="CSV generated by extract-local-nwb",
    )
    phase_parser.add_argument("--bounds", required=True)
    phase_parser.add_argument(
        "--parent-models",
        required=True,
        help="Step-1 elite table containing diverse fast Na/K parents",
    )
    phase_parser.add_argument(
        "--n-parents",
        type=int,
        default=2,
    )
    phase_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    phase_parser.add_argument("--cell-id", required=True)
    phase_parser.add_argument("--temperature", type=float, required=True)
    phase_parser.add_argument(
        "--screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    phase_parser.add_argument("--population-size", type=int, default=8)
    phase_parser.add_argument("--generations", type=int, default=3)
    phase_parser.add_argument("--workers", type=int, default=1)
    phase_parser.add_argument("--seed", type=int, default=42)
    phase_parser.add_argument("--sigma", type=float, default=0.40)
    phase_parser.add_argument("--dt", type=float, default=0.025)
    phase_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    phase_parser.add_argument(
        "--phase-step-duration-ms",
        type=float,
        default=300.0,
    )
    phase_parser.add_argument(
        "--phase-grid-points",
        type=int,
        default=64,
    )
    phase_parser.add_argument(
        "--spline-smoothing",
        type=float,
        default=0.002,
    )
    phase_parser.add_argument("--gna-gk-factor", type=float, default=1.5)
    phase_parser.add_argument(
        "--output-dir",
        default="outputs/phase_shape_experiment",
    )
    phase_parser.set_defaults(func=_phase_shape_experiment)

    pareto_parser = subparsers.add_parser(
        "phase-pareto-experiment",
        help="Compare nested sodium-activation mechanisms with NSGA-II",
    )
    pareto_parser.add_argument(
        "raw_spike_cycles",
        help="CSV generated by extract-local-nwb",
    )
    pareto_parser.add_argument("--bounds", required=True)
    pareto_parser.add_argument(
        "--baseline-model",
        required=True,
        help="Best soma-AIS model from phase-shape-experiment",
    )
    pareto_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    pareto_parser.add_argument("--cell-id", required=True)
    pareto_parser.add_argument("--temperature", type=float, required=True)
    pareto_parser.add_argument(
        "--screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    pareto_parser.add_argument("--population-size", type=int, default=12)
    pareto_parser.add_argument("--generations", type=int, default=4)
    pareto_parser.add_argument("--workers", type=int, default=1)
    pareto_parser.add_argument("--seed", type=int, default=42)
    pareto_parser.add_argument(
        "--initialization-fraction",
        type=float,
        default=0.08,
    )
    pareto_parser.add_argument("--dt", type=float, default=0.025)
    pareto_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    pareto_parser.add_argument(
        "--phase-step-duration-ms",
        type=float,
        default=300.0,
    )
    pareto_parser.add_argument(
        "--phase-grid-points",
        type=int,
        default=64,
    )
    pareto_parser.add_argument(
        "--spline-smoothing",
        type=float,
        default=0.002,
    )
    pareto_parser.add_argument("--gna-gk-factor", type=float, default=1.5)
    pareto_parser.add_argument(
        "--output-dir",
        default="outputs/phase_pareto_experiment",
    )
    pareto_parser.set_defaults(func=_phase_pareto_experiment)

    direct_phase_parser = subparsers.add_parser(
        "direct-kinetics-phase-fit",
        help=(
            "Fit direct x_inf/tau gates in shape, physical-scale, and "
            "joint trust-region stages"
        ),
    )
    direct_phase_parser.add_argument(
        "raw_spike_cycles",
        help="CSV generated by extract-local-nwb",
    )
    direct_phase_parser.add_argument(
        "--baseline-model",
        required=True,
        help="Best soma-AIS model used to initialize the direct gate curves",
    )
    direct_phase_parser.add_argument(
        "--biological-dataset",
        choices=("gouwens_visp", "scala_room_temperature"),
        required=True,
    )
    direct_phase_parser.add_argument("--cell-id", required=True)
    direct_phase_parser.add_argument(
        "--temperature",
        type=float,
        required=True,
    )
    direct_phase_parser.add_argument(
        "--screen-preset",
        choices=("scala", "gouwens"),
        required=True,
    )
    direct_phase_parser.add_argument(
        "--shape-population-size",
        type=int,
        default=10,
    )
    direct_phase_parser.add_argument(
        "--shape-generations",
        type=int,
        default=3,
    )
    direct_phase_parser.add_argument("--shape-elites", type=int, default=2)
    direct_phase_parser.add_argument(
        "--scale-population-size",
        type=int,
        default=8,
    )
    direct_phase_parser.add_argument(
        "--scale-generations",
        type=int,
        default=2,
    )
    direct_phase_parser.add_argument(
        "--final-population-size",
        type=int,
        default=10,
    )
    direct_phase_parser.add_argument(
        "--final-generations",
        type=int,
        default=2,
    )
    direct_phase_parser.add_argument("--workers", type=int, default=1)
    direct_phase_parser.add_argument("--seed", type=int, default=42)
    direct_phase_parser.add_argument("--sigma", type=float, default=0.18)
    direct_phase_parser.add_argument("--dt", type=float, default=0.05)
    direct_phase_parser.add_argument(
        "--rheobase-tolerance-pa",
        type=float,
        default=2.0,
    )
    direct_phase_parser.add_argument(
        "--phase-step-duration-ms",
        type=float,
        default=300.0,
    )
    direct_phase_parser.add_argument(
        "--phase-grid-points",
        type=int,
        default=64,
    )
    direct_phase_parser.add_argument(
        "--spline-smoothing",
        type=float,
        default=0.002,
    )
    direct_phase_parser.add_argument(
        "--velocity-tolerance-fraction",
        type=float,
        default=0.10,
    )
    direct_phase_parser.add_argument(
        "--voltage-tolerance-mv",
        type=float,
        default=4.0,
    )
    direct_phase_parser.add_argument(
        "--trust-fraction",
        type=float,
        default=0.20,
    )
    direct_phase_parser.add_argument(
        "--gna-gk-factor",
        type=float,
        default=1.8,
    )
    direct_phase_parser.add_argument(
        "--output-dir",
        default="outputs/direct_kinetics_phase_fit",
    )
    direct_phase_parser.set_defaults(func=_direct_phase_fit)

    calibration_parser = subparsers.add_parser(
        "calibrate-prior",
        help="Propose a padded prior from models nearest to biological recordings",
    )
    calibration_parser.add_argument("models", help="Accepted HH model dataset CSV")
    calibration_parser.add_argument(
        "model_plausibility",
        help="model_plausibility.csv generated by the coverage command",
    )
    calibration_parser.add_argument(
        "--parameter-group",
        choices=("static", "kinetic", "all"),
        default="static",
    )
    calibration_parser.add_argument("--minimum-models", type=int, default=5)
    calibration_parser.add_argument(
        "--kinetic-expansion",
        type=float,
        default=1.0,
        help="Expand default kinetic windows around the canonical origin",
    )
    calibration_parser.add_argument(
        "--output",
        default="outputs/calibrated_prior.json",
    )
    calibration_parser.set_defaults(func=_calibrate_prior)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

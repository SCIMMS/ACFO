from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}


def load_openmm() -> SimpleNamespace:
    try:
        import openmm as mm
        from openmm import unit
        from openmm.app import (
            DCDReporter,
            ForceField,
            HBonds,
            Modeller,
            PDBFile,
            PME,
            Simulation,
            StateDataReporter,
            Topology,
        )
        from openmm.app.element import hydrogen, oxygen
    except ImportError as exc:  # pragma: no cover - depends on server env
        raise SystemExit(
            "OpenMM is required. On the lab server, run this inside the openmm conda env."
        ) from exc

    return SimpleNamespace(
        mm=mm,
        unit=unit,
        DCDReporter=DCDReporter,
        ForceField=ForceField,
        HBonds=HBonds,
        Modeller=Modeller,
        PDBFile=PDBFile,
        PME=PME,
        Simulation=Simulation,
        StateDataReporter=StateDataReporter,
        Topology=Topology,
        hydrogen=hydrogen,
        oxygen=oxygen,
    )


def ps_to_steps(ps: float, timestep_fs: float) -> int:
    if ps <= 0:
        return 0
    return int(round(ps * 1000.0 / timestep_fs))


def make_seed_water_modeller(omm: SimpleNamespace, box_nm: float):
    """Create a one-water TIP3P seed so Modeller.addSolvent never sees emptiness."""

    mm = omm.mm
    unit = omm.unit
    topology = omm.Topology()
    chain = topology.addChain("A")
    residue = topology.addResidue("HOH", chain)
    oxygen_atom = topology.addAtom("O", omm.oxygen, residue)
    h1 = topology.addAtom("H1", omm.hydrogen, residue)
    h2 = topology.addAtom("H2", omm.hydrogen, residue)
    topology.addBond(oxygen_atom, h1)
    topology.addBond(oxygen_atom, h2)
    topology.setPeriodicBoxVectors(
        (
            mm.Vec3(box_nm, 0.0, 0.0),
            mm.Vec3(0.0, box_nm, 0.0),
            mm.Vec3(0.0, 0.0, box_nm),
        )
        * unit.nanometer
    )

    # TIP3P gas-phase geometry in nm. The later minimization/equilibration relaxes it.
    positions = [
        mm.Vec3(0.000000, 0.000000, 0.000000),
        mm.Vec3(0.095720, 0.000000, 0.000000),
        mm.Vec3(-0.023998, 0.092663, 0.000000),
    ] * unit.nanometer
    return omm.Modeller(topology, positions)


def load_modeller_or_seed(
    omm: SimpleNamespace,
    input_pdb: Path | None,
    box_nm: float,
    *,
    allow_seed_fallback: bool,
):
    if input_pdb is None:
        return make_seed_water_modeller(omm, box_nm), "seed_tip3p_water"

    try:
        pdb = omm.PDBFile(str(input_pdb))
        n_atoms = sum(1 for _ in pdb.topology.atoms())
        if n_atoms > 0:
            return omm.Modeller(pdb.topology, pdb.positions), f"input_pdb:{input_pdb}"
        message = f"{input_pdb} contains no atoms"
    except Exception as exc:
        message = f"failed to load {input_pdb}: {exc}"

    if not allow_seed_fallback:
        raise ValueError(message)
    print(f"[water-box] {message}; using one TIP3P seed water instead", file=sys.stderr)
    return make_seed_water_modeller(omm, box_nm), "seed_tip3p_water_after_empty_pdb"


def choose_platform(
    omm: SimpleNamespace,
    platform_name: str,
    *,
    precision: str,
    device_index: str | None,
):
    mm = omm.mm
    props: dict[str, str] = {}
    candidates = ["CUDA", "OpenCL", "CPU"] if platform_name == "auto" else [platform_name]
    last_error: Exception | None = None
    for name in candidates:
        try:
            platform = mm.Platform.getPlatformByName(name)
        except Exception as exc:
            last_error = exc
            continue
        if name == "CUDA":
            props["Precision"] = precision
            if device_index is not None:
                props["DeviceIndex"] = str(device_index)
        return platform, props
    raise RuntimeError(f"could not select OpenMM platform from {candidates}: {last_error}")


def count_water_residues(topology) -> int:
    return sum(1 for residue in topology.residues() if residue.name.upper() in WATER_RESNAMES)


def topology_elements(topology) -> np.ndarray:
    return np.asarray(
        [
            atom.element.symbol if atom.element is not None else atom.name
            for atom in topology.atoms()
        ],
        dtype="<U4",
    )


def box_vectors_nm(state, unit) -> np.ndarray:
    vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    return np.asarray(vectors, dtype=np.float64)


def positions_nm(state, unit) -> np.ndarray:
    return np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))


def write_npz(path: Path, topology, state, unit, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=positions_nm(state, unit),
        elements=topology_elements(topology),
        box_vectors=box_vectors_nm(state, unit),
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"), sort_keys=True)),
    )


def write_pdb(path: Path, omm: SimpleNamespace, topology, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        omm.PDBFile.writeFile(topology, state.getPositions(), fh)


def build_water_box(args: argparse.Namespace) -> dict:
    omm = load_openmm()
    mm = omm.mm
    unit = omm.unit
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    modeller, input_source = load_modeller_or_seed(
        omm,
        args.input_pdb,
        args.box_nm,
        allow_seed_fallback=not args.no_seed_fallback,
    )
    forcefield = omm.ForceField(args.water_ff)
    modeller.addSolvent(
        forcefield,
        model=args.water_model,
        boxSize=mm.Vec3(args.box_nm, args.box_nm, args.box_nm) * unit.nanometer,
        neutralize=False,
    )

    solvated_pdb = out_dir / f"{args.prefix}_solvated_initial.pdb"
    with solvated_pdb.open("w", encoding="utf-8") as fh:
        omm.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=omm.PME,
        nonbondedCutoff=args.nonbonded_cutoff_nm * unit.nanometer,
        constraints=omm.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=args.ewald_error_tolerance,
    )
    if args.pressure_bar > 0.0:
        barostat = mm.MonteCarloBarostat(
            args.pressure_bar * unit.bar,
            args.temperature_k * unit.kelvin,
            args.barostat_interval,
        )
        barostat.setRandomNumberSeed(args.seed + 1)
        system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_per_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(args.seed)
    platform, platform_props = choose_platform(
        omm,
        args.platform,
        precision=args.cuda_precision,
        device_index=args.device_index,
    )
    simulation = omm.Simulation(
        modeller.topology,
        system,
        integrator,
        platform,
        platform_props,
    )
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed)

    print(
        "[water-box] built "
        f"{sum(1 for _ in modeller.topology.atoms())} atoms, "
        f"{count_water_residues(modeller.topology)} waters on {platform.getName()}",
        flush=True,
    )
    if args.minimize:
        print("[water-box] minimizing", flush=True)
        simulation.minimizeEnergy(maxIterations=args.minimize_iterations)

    equilibration_steps = ps_to_steps(args.equilibration_ps, args.timestep_fs)
    if equilibration_steps:
        print(f"[water-box] equilibrating {equilibration_steps} steps", flush=True)
        simulation.step(equilibration_steps)

    production_steps = ps_to_steps(args.production_ps, args.timestep_fs)
    report_steps = max(1, ps_to_steps(args.report_interval_ps, args.timestep_fs))
    dcd_path = out_dir / f"{args.prefix}_trajectory.dcd"
    log_path = out_dir / f"{args.prefix}_state.csv"
    if production_steps:
        simulation.reporters.append(
            omm.DCDReporter(str(dcd_path), report_steps, enforcePeriodicBox=True)
        )
        simulation.reporters.append(
            omm.StateDataReporter(
                str(log_path),
                report_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                density=True,
                speed=True,
                separator=",",
            )
        )
        print(f"[water-box] production {production_steps} steps", flush=True)
        simulation.step(production_steps)

    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
        enforcePeriodicBox=True,
    )
    modeller.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    metadata = {
        "box_nm_requested": args.box_nm,
        "equilibration_ps": args.equilibration_ps,
        "final_box_vectors_nm": box_vectors_nm(state, unit).tolist(),
        "friction_per_ps": args.friction_per_ps,
        "input_source": input_source,
        "kind": "openmm_tip3p_water_box",
        "n_atoms": sum(1 for _ in modeller.topology.atoms()),
        "n_waters": count_water_residues(modeller.topology),
        "nonbonded_cutoff_nm": args.nonbonded_cutoff_nm,
        "openmm_platform": platform.getName(),
        "openmm_version": getattr(mm.version, "version", "unknown"),
        "pressure_bar": args.pressure_bar,
        "production_ps": args.production_ps,
        "report_interval_ps": args.report_interval_ps,
        "seed": args.seed,
        "temperature_k": args.temperature_k,
        "timestep_fs": args.timestep_fs,
        "water_ff": args.water_ff,
        "water_model": args.water_model,
    }
    metadata_path = out_dir / f"{args.prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    write_pdb(out_dir / f"{args.prefix}_final.pdb", omm, modeller.topology, state)
    write_npz(out_dir / f"{args.prefix}_final.npz", modeller.topology, state, unit, metadata)
    checkpoint_path = out_dir / f"{args.prefix}_checkpoint.chk"
    with checkpoint_path.open("wb") as fh:
        fh.write(simulation.context.createCheckpoint())
    state_xml = mm.XmlSerializer.serialize(state)
    (out_dir / f"{args.prefix}_state.xml").write_text(state_xml, encoding="utf-8")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and simulate a pure TIP3P water box with OpenMM. "
            "If no input PDB is provided, or if the PDB is empty, a single seed "
            "water molecule is used before Modeller.addSolvent()."
        )
    )
    parser.add_argument("--box-nm", type=float, default=8.0)
    parser.add_argument("--input-pdb", type=Path)
    parser.add_argument("--no-seed-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--water-ff", default="tip3p.xml")
    parser.add_argument("--water-model", default="tip3p")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    parser.add_argument("--friction-per-ps", type=float, default=1.0)
    parser.add_argument("--equilibration-ps", type=float, default=200.0)
    parser.add_argument("--production-ps", type=float, default=200.0)
    parser.add_argument("--report-interval-ps", type=float, default=10.0)
    parser.add_argument("--nonbonded-cutoff-nm", type=float, default=1.0)
    parser.add_argument("--ewald-error-tolerance", type=float, default=5e-4)
    parser.add_argument("--barostat-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--platform", choices=["CUDA", "OpenCL", "CPU", "auto"], default="CUDA")
    parser.add_argument("--cuda-precision", choices=["single", "mixed", "double"], default="mixed")
    parser.add_argument("--device-index")
    parser.add_argument("--minimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--minimize-iterations", type=int, default=500)
    args = parser.parse_args()

    if args.box_nm <= 0:
        raise ValueError("--box-nm must be positive")
    if args.timestep_fs <= 0:
        raise ValueError("--timestep-fs must be positive")
    if args.report_interval_ps <= 0:
        raise ValueError("--report-interval-ps must be positive")
    if args.prefix is None:
        box_tag = f"{args.box_nm:g}".replace(".", "p")
        args.prefix = f"water_tip3p_{box_tag}nm"
    if args.output_dir is None:
        args.output_dir = Path("runs") / args.prefix
    return args


def main() -> None:
    metadata = build_water_box(parse_args())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

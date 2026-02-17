from pathlib import Path

from geometry import MOCSolver


def main() -> None:
    out = Path("out/moc")
    out.mkdir(parents=True, exist_ok=True)

    solver = MOCSolver(machExit=2.4, pressureRatio=0.09)
    geom = solver.generateGeometry(
        {
            "throat_y": 0.02,
            "exit_y": 0.062,
            "length": 0.22,
            "n_points": 180,
        }
    )
    geom.metadata["id"] = "moc_only"
    geom.exportGeo(str(out / "moc_geometry.csv"))
    geom.plotProfile(str(out / "moc_geometry.png"))


if __name__ == "__main__":
    main()

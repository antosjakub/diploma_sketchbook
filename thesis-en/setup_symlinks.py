from pathlib import Path


names = ["PINN", "SG", "HeatEq", "Smoluchowski"]
fig_dir = "figures"
LINK_CHAPTERS = True
LINK_FIGURES = True


def ensure_directory_exists(directory: Path) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")


def remove_existing_symlink(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not path.is_symlink():
        raise RuntimeError(f"Refusing to remove non-symlink path: {path}")
    print(f"Removing symlink: {path}")
    path.unlink()


def create_symlink(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Symlink source does not exist: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}")
    print(f"Creating symlink: {destination} -> {source}")
    destination.symlink_to(source)


if LINK_CHAPTERS:
    print("===== 2. Link chapters (PINN/thesis-en/chapter.tex to chapter_pinn.tex, etc.)")
    for name in names:
        chapter_tex = Path(f"chapter_{name.lower()}.tex")
        chapter_source = Path(f"../{name}/thesis-en/chapter.tex")
        remove_existing_symlink(chapter_tex)
        create_symlink(chapter_source, chapter_tex)


if LINK_FIGURES:
    print("===== 3. Link figures")

    print("-- 3a: Remove all symlinks")
    figures_directory = Path(fig_dir)
    ensure_directory_exists(figures_directory)
    for entry in figures_directory.iterdir():
        if entry.is_symlink():
            remove_existing_symlink(entry)

    print("-- 3b: Populate figures/ with PINN/thesis-en/figures/*, etc.")
    seen_destinations = set()
    for name in names:
        source_directory = Path(f"../{name}/thesis-en/{fig_dir}")
        ensure_directory_exists(source_directory)
        for entry in source_directory.iterdir():
            destination = figures_directory / entry.name
            if destination.name in seen_destinations:
                raise RuntimeError(
                    f"Duplicate figure filename detected: {destination.name}"
                )
            seen_destinations.add(destination.name)
            create_symlink(entry, destination)


import subprocess
from pathlib import Path

names = ["PINN"]
fig_dir = 'figures'
LINK_CHAPTERS = True
LINK_FIGURES =  True
SETUP_BIB =     True # concentrate all bib files into one


if SETUP_BIB:
    print("===== 1. Concentrate all bibs into one")
    info = [
        '%',
        '%  An example of a bibliographical database in BibTeX format,',
        '%  which is used by biblatex to create the list of referenced works.',
        '%',
        '%  Academic search engines and software for maintenance of bibliography',
        '%  often supports exporting records in BibTeX format. We recommend:',
        '%',
        '%    - Google Scholar (https://scholar.google.com/)',
        '%    - JabRef (https://www.jabref.org/)',
        '%    - zoterobib (https://zbib.org/)',
        '%',
        '%  BEWARE:',
        '%',
        '%    *  If a name contains a capital letter, which must be kept such,',
        '%       use curly brackets ({T}hailand, {HIV}).',
        '%',
        '%  ===========================================================================',
        '',
        ''
    ]
    all_lines = [f'{l}\n' for l in info]
    filename = "bibliography.bib"
    for name in names:
        filename_rel = f'../{name}/thesis-en/{filename}'
        with open(filename_rel, "r") as f:
            lines = f.readlines()
            all_lines.append(f"% --- {name}")
            all_lines.extend(lines)
            all_lines.append("\n")
    #text = '\n'.join(all_lines)
    with open(filename, "w") as f:
        f.writelines(all_lines)


if LINK_CHAPTERS:
    print("===== 2. Link chapters (PINN/thesis-en/chapter.tex to chapter_pinn.tex, etc.)")
    for name in names:
        name_lower = name.lower()
        chapter_tex = f'chapter_{name_lower}.tex'
        print(f"Creating symlink: {chapter_tex}")
        command = ['rm', chapter_tex]
        result = subprocess.run(command, capture_output=True, text=True)
        command = ['ln', '-s', f'../{name}/thesis-en/chapter.tex', chapter_tex]
        result = subprocess.run(command, capture_output=True, text=True)
        #print("STDOUT:", result.stdout)
        #print("STDERR:", result.stderr)
        #print("Return code:", result.returncode)


if LINK_FIGURES:
    print("===== 3. Link figures")

    print("-- 3a: Remove all symlinks")
    directory = Path(fig_dir)
    for entry in directory.iterdir():
        is_link = entry.is_symlink()
        file = entry.name
        #print(file, "-> symlink" if is_link else "-> regular")
        if is_link:
            print(f'Removing symlink: {file}')
            command = ['rm', f'{fig_dir}/{file}']
            result = subprocess.run(command, capture_output=True, text=True)

    print("-- 3b: Populate figures/ with PINN/thesis-en/figures/*, etc.")
    for name in names:
        directory = Path(f'../{name}/thesis-en/{fig_dir}')
        for entry in directory.iterdir():
            file = entry.name
            print(f'Creating a symlink: {file}')
            command = ['ln', '-s', str(entry), f'{fig_dir}/{file}']
            result = subprocess.run(command, capture_output=True, text=True)
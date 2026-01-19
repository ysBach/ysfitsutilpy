import os
import re

def fix_content(s):
    # 1. Fix double namespaces (caused by previous script)
    # pandas.~pandas.DataFrame -> ~pandas.DataFrame
    # Ref: keys were DataFrame, Series, ndarray, Path, CCDData

    corrections = [
        (r'pandas\.~pandas\.DataFrame', r'~pandas.DataFrame'),
        (r'pandas\.~pandas\.Series', r'~pandas.Series'),
        (r'numpy\.~numpy\.ndarray', r'~numpy.ndarray'),
        (r'pathlib\.~pathlib\.Path', r'~pathlib.Path'),
        (r'astropy\.nddata\.~astropy\.nddata\.CCDData', r'~astropy.nddata.CCDData'),
    ]

    for pat, repl in corrections:
        s = re.sub(pat, repl, s)

    # 2. Linkify fully qualified names (if not already tilded)
    # pandas.DataFrame -> ~pandas.DataFrame
    # Use negative lookbehind for ~

    fq_types = [
        "pandas.DataFrame",
        "pandas.Series",
        "numpy.ndarray",
        "astropy.table.Table",
        "astropy.io.fits.PrimaryHDU",
        "astropy.io.fits.ImageHDU",
        "astropy.io.fits.HDUList",
        "astropy.io.fits.Header",
        "astropy.wcs.WCS",
        "astropy.coordinates.SkyCoord",
        "astropy.time.Time",
    ]

    for fqt in fq_types:
        # Regex: (?<![~`])\bFQT\b
        # Match FQT not preceded by ~ or `
        pat = r'(?<![~`])\b' + re.escape(fqt) + r'\b'
        s = re.sub(pat, r'~' + fqt, s)

    return s

def process_file(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    fixed = fix_content(content)

    if fixed != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(fixed)

def main():
    target_dir = "ysfitsutilpy"
    for root, dirs, files in os.walk(target_dir):
        for name in files:
            if name.endswith(".py"):
                print(f"Processing {name}...")
                process_file(os.path.join(root, name))

if __name__ == "__main__":
    main()

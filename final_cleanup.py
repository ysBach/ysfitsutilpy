import os
import re

def fix_content(s):
    # 1. Fix double tildes
    while '~~' in s:
        s = s.replace('~~', '~')

    # 2. Fix fits.Type -> ~astropy.io.fits.Type
    # Regex word boundary
    # fits.PrimaryHDU -> ~astropy.io.fits.PrimaryHDU
    # But only if not preceded by io. (astropy.io.fits.PrimaryHDU)
    # Check if 'fits.' is preceded by nothing or space or comma etc?

    fits_types = ["PrimaryHDU", "ImageHDU", "HDUList", "Header"]
    for ft in fits_types:
        # replace fits.Type with ~astropy.io.fits.Type
        pat = r'\bfits\.' + ft + r'\b'
        s = re.sub(pat, r'~astropy.io.fits.' + ft, s)

    # 3. Add space after comma if followed by ~
    # ,~ -> , ~
    s = s.replace(',~', ', ~')

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

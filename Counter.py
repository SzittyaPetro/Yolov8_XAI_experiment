import os
from pathlib import Path

purpose = ('test')


def collect_png_files(directory):
    """
    Collect all .png files in the given directory and its subdirectories.
    """
    global purpose
    png_files = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            for root2, dirs2, files2 in os.walk(Path(root)/dir):
                for file in files2:
                    if file.endswith('.png'):
                        png_files.append(os.path.join(directory+"/"+dir+"/", file))
    return png_files


def write_to_txt(filename, png_files):
    """
    Write the list of .png files to a .txt file.
    """
    with open(filename, 'w') as f:
        for png_file in png_files:
            f.write(png_file + '\n')


# Mappa, amelyet be szeretnénk járni
directory = 'D:/Uni/Yolov8/data/gtFine/images/'+purpose

# Összegyűjti az összes .png fájlt a mappában és az alkönyvtárakban
png_files = collect_png_files(directory)

# Kiírja a .png fájlok listáját egy .txt fájlba
output_file = str(Path(directory)) + ".txt"
write_to_txt(output_file, png_files)

print(f"All .png files have been collected and written to {output_file}.")

from onmt.utils.misc import read_lines
base_path = "data/text_simplification/OneStopEnglish/"
file_paths = [(base_path + "ADV-ELE.txt"), (base_path + "ADV-INT.txt")]
levels = [1, 2]

for level, file_path in zip(levels, file_paths):
    lines = read_lines(file_path)
    src_file_path = base_path + "src." + str(level)
    src_file = open(src_file_path, "wb")
    tgt_file_path = base_path + "tgt." + str(level)
    tgt_file = open(tgt_file_path, "wb")
    for i in range(0, len(lines), 3):
        src_file.write(lines[i])
        tgt_file.write(lines[i+1])

    src_file.close()
    tgt_file.close()



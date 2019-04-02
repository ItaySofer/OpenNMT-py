import json


def take_num(s):
    return int(s[-1])


# Create doc to grade levels mapping
with open('./data/text_simplification/Newsela/newsela_articles_20150302.5versions.sents.json', 'r', encoding='utf8') as f:
    newsela_json = json.load(f)

doc_grade_levels = [[int(article['grade_level']) for article in subject['articles']] for subject in newsela_json]
doc_grade_levels = [sorted(grade_levels, reverse=True) for grade_levels in doc_grade_levels]


# Create src and tgt sentences with their grade levels
with open('./data/text_simplification/Newsela/newsela_articles_20150302.aligned.sents.txt', 'r', encoding='utf8') as f:
    raw_entries = f.readlines()

entries = [entry.split('\t') for entry in raw_entries]
entries = [{"doc": take_num(entry[0]) - 1,
            "src_version": take_num(entry[1]),
            "tgt_version": take_num(entry[2]),
            "src_sent": entry[3].strip(),
            "tgt_sent": entry[4].strip()}
           for entry in entries]
entries = [{"src_grade_level": doc_grade_levels[entry['doc']][entry['src_version']],
            "tgt_grade_level": doc_grade_levels[entry['doc']][entry['tgt_version']],
            "src_sent": entry['src_sent'],
            "tgt_sent": entry['tgt_sent']}
           for entry in entries]


# Write sentences to src and tgt files, according to tgt grade level
base_path = "./data/text_simplification/Newsela/"
files = {}
for level in range(2, 12):
    files[level] = {"src_file": open(base_path + "src." + str(level), "w"),
                    "tgt_file": open(base_path + "tgt." + str(level), "w")}


num_of_sent_per_tgt_level = {i: 0 for i in range(2, 12)}
for entry in entries:
    tgt_level = entry['tgt_grade_level']
    src_sent = entry['src_sent']
    tgt_sent = entry['tgt_sent']

    src_file = files[tgt_level]['src_file']
    tgt_file = files[tgt_level]['tgt_file']

    src_file.writelines(src_sent + '\n')
    tgt_file.write(tgt_sent + '\n')

    num_of_sent_per_tgt_level[tgt_level] = num_of_sent_per_tgt_level[tgt_level] + 1

print("number of sentences per target grade level: " + str(num_of_sent_per_tgt_level))

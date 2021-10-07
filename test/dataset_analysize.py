import json

train_path = '../dataset/train.json'
data = []
with open(train_path, 'r') as fread:
    for line in fread.readlines():
        data.append(json.loads(line))
# 字段
data[0].keys()
# dict_keys(['pid', 'is_classical', 'title', 'author', 'paragraphs', 'qas'])


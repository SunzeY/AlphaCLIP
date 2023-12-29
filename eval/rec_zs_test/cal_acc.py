import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, default='refcoco_val')

args = parser.parse_args()

name = args.name
print(name)
count = 0
all_count = 0
for i in range(8):
    pth = f'output/{name}_count_{i}.json'
    acc = json.load(open(pth, 'r'))
    a_list = acc.split()
    a, b = a_list[0], a_list[1]
    count += int(a)
    all_count += int(b)

print(float(count) / float(all_count))
import csv
import pandas as pd  # not key to functionality of kernel

sub_files = [
    '../input/ensemb/submission_1.csv',
    '../input/ensemb/submission_0.6_standard_0.4_boostrap(0.85).csv',
    '../input/ensemb/submission_0.6_standard_0.4_boostrap(0.85).csv',

]

# Weights of the individual subs
sub_weight = [
    0.842 ** 2,
    0.896 ** 2,
    0.896 ** 2,

]
# 15
Hlabel = 'Image'
Htarget = 'Id'
npt = 6  # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = (1 / (i + 1))

print(place_weights)

lg = len(sub_files)
sub = [None] * lg
for i, file in enumerate(sub_files):
    ## input files ##
    print("Reading {}: w={} - {}".format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file, "r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

## output file ##
out = open("sub_siamese_ens20.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel, Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt, 0) + (place_weights[ind] * sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:5]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()
import matplotlib.pyplot as plt

types = ["norm", "bckg", "artf", "null", "nneo", "infl", "susp", "dcis", "indc"]
colors = ["lightpink", "peachpuff", "#CBC3DB", "#BAD9BB", "lightblue", "thistle", "#BED4E9", "pink", "#C5CDBA"]
classified = ["norm", "bckg", "artf", "null", "nneo", "infl", "susp", "dcis", "indc", "norm", "bckg", "artf", "null"]

label_count = {}

for label in classified:
    if label in label_count:
        label_count[label] += 1
    else:
        label_count[label] = 1

for t in types:
    if t not in label_count:
        label_count[t] = 0

labels = list(label_count.keys())
counts = list(label_count.values())

plt.bar(labels, counts, color=colors)
plt.xlabel('Label Types')
plt.ylabel('Count')
plt.title('Classification of Biopsy Slides')

plt.savefig('histogram.jpg')
plt.show()
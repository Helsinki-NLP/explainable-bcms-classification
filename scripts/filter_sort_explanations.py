import json, sys, statistics


def process(filename, topn):
	data = {}
	f = open(filename, 'r')
	for line in f:
		wordinfo = json.loads(line)
		attested_classes = [k for k in wordinfo if k != "word" and len(wordinfo[k]) > 0]
		if len(attested_classes) != 1:
			#print("Skip", wordinfo["word"], "(explanation for more than one class)")
			continue
		non_hapax_classes = [k for k in wordinfo if k != "word" and len(wordinfo[k]) > 1]
		if len(non_hapax_classes) == 0:
			#print("Skip", wordinfo["word"], "(hapax)")
			continue
		pred_class = non_hapax_classes[0]
		if pred_class not in data:
			data[pred_class] = {}
		average_diff = statistics.mean(wordinfo[pred_class])
		data[pred_class][wordinfo["word"]] = average_diff
	f.close()

	#print("Words after filtering:", len(data))
	for cl in sorted(data):
		print("***", cl, "***")
		sorted_data = sorted(data[cl], key=data[cl].get, reverse=True)
		for word in sorted_data[:topn]:
			print(f"{word}\t{data[cl][word]:.3f}")
		print()


if __name__ == "__main__":
	filename = sys.argv[1]
	topn = int(sys.argv[2])
	process(filename, topn)

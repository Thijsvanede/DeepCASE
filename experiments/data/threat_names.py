import json

if __name__ == "__main__":
    with open('../../../data/preprocessed.csv.encoding.json') as infile:
        threat_names = json.load(infile).get('threat_name')

    threat_names.append("")

    for left, right in zip(threat_names[::2], threat_names[1::2]):
        print("{:50} & {:50} \\\\".format(left, right))

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    args = parser.parse_args()

    with open(args.file) as infile:
        days = args.file.split('_')[-1].split('.')[0]
        with open("{}_days.result".format(days), 'w') as outfile:
            for line in infile.readlines():
                if "it/s" not in line and "s/it" not in line:
                    outfile.write(line)

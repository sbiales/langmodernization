from lxml import etree
from io import StringIO
import os
import argparse
from tqdm import tqdm

def main(args):
    if args.corpus == 'clmet':
        process_clmet(args.corpus_path, args.data_dir)

def process_clmet(path, data_dir):
    print('Processing CLMET corpus txt files')
    # Separate the corpus into 3 lists, one for each time period
    p1710_1780 = []
    p1780_1850 = []
    p1850_1920 = []

    for filename in tqdm(os.listdir(path)):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            if "&" in content:
                        content = content.replace("&", "&#38;")
            content = '<xml>' + content + '</xml>' # Needed to make a parsable xml doc
            try:
                tree = etree.parse(StringIO(content))
            except Exception as e:
                print('\nError processing file ' + filename +': ' + repr(e))

            r = tree.xpath('/xml/period')
            period = r[0].text
            r = tree.xpath('/xml/text/p')
            lines = []
            for line in r:
                lines.extend([l for l in line.text.strip().split('\n') if l != ''])
            if period == '1710-1780':
                p1710_1780.extend(lines)
            elif period == '1780-1850':
                p1780_1850.extend(lines)
            elif period == '1850-1920':
                p1850_1920.extend(lines)
            else:
                print('File ', filename, ': Period ', period, 'does not fit into a category')

    # Save the corpus files
    outpath = os.path.join(os.getcwd(), data_dir, 'clmet')
    if not os.path.exists(os.path.join(os.getcwd(), data_dir)):
        os.mkdir(os.path.join(os.getcwd(), data_dir))
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    with open(os.path.join(outpath, "1710_1780.txt"), "w", encoding='utf-8') as outfile:
        outfile.write("\n".join(p1710_1780))

    with open(os.path.join(outpath, "1780_1850.txt"), "w", encoding='utf-8') as outfile:
        outfile.write("\n".join(p1780_1850))

    with open(os.path.join(outpath, "1850_1920.txt"), "w", encoding='utf-8') as outfile:
        outfile.write("\n".join(p1850_1920))
    
    print('Files saved to', outpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--corpus', type=str, default='clmet')
    parser.add_argument('-cp', '--corpus_path', type=str, default="C:\\Users\\siena\\repositories\\clmet\\clmet\\corpus\\txt\\plain")
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()

    assert args.corpus in ['clmet', 'ced']

    main(args)
import re


def main():
    # Split the original dataset into parallel English and French txts
    eng_sentences = []
    fra_sentences = []
    with open('data/raw/fra.txt') as f:
        for line in f:
            line = re.split(r'\t+', line)
            eng_sentences.append(line[0])
            fra_sentences.append(line[1])

    with open('data/raw/english.txt', 'w') as f:
        for line in eng_sentences:
            f.write(line + '\n')
    with open('data/raw/french.txt', 'w') as f:
        for line in fra_sentences:
            f.write(line + '\n')


if __name__ == "__main__":
    main()

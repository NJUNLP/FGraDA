import argparse
import json


def main(args):
    test_zh = [line.strip() for line in open(
        args['test_zh_path'], 'r').readlines()]
    test_en = [line.strip() for line in open(
        args['test_en_path'], 'r').readlines()]
    dict_zh = [line.strip() for line in open(
        args['dict_zh_path'], 'r').readlines()]
    dict_en = [line.strip() for line in open(
        args['dict_en_path'], 'r').readlines()]

    cons_en = []
    for zh_line, en_line in zip(test_zh, test_en):
        cons = []
        if args['build_empty_constraint']:
            pass
        else:
            for zh_phrase, en_phrase in zip(dict_zh, dict_en):
                if ' {} '.format(zh_phrase) in ' {} '.format(zh_line) and \
                        ' {} '.format(en_phrase) in ' {} '.format(en_line):
                    cons.append(en_phrase.split())

            if len(cons) > 0:
                remove_cons = []
                for i in range(len(cons)):
                    for j in range(i+1, len(cons)):
                        if ' '.join(cons[i]) in ' '.join(cons[j]):
                            remove_cons.append(cons[i])
                            break
                if len(remove_cons) > 0:
                    for _ in remove_cons:
                        cons.remove(_)

        cons_en.append(cons)

    json.dump(cons_en, open(args['cons_en_path'], 'w'), indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_zh_path', type=str,
                        help='path to Chinese text (split to subwords)')
    parser.add_argument('--test_en_path', type=str,
                        help='path to English text (split to subwords)')
    parser.add_argument('--dict_zh_path', type=str,
                        help='path to Chinese phrases in bilingual dictionary (split to subword)')
    parser.add_argument('--dict_en_path', type=str,
                        help='path to English phrases in bilingual dictionary (split to subword)')
    parser.add_argument('--cons_en_path', type=str,
                        help='path to save the constraint information')
    parser.add_argument('--build_empty_constraint', action='store_true')

    args = parser.parse_args()
    args = vars(args)

    main(args)

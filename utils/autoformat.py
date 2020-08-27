import os
import json

INFO_FNAME = './AUTOFORMAT.json'
INFO_DICT = dict()


def load_info_dict():
    global INFO_DICT
    if os.path.exists(INFO_FNAME):
        with open(INFO_FNAME, 'r') as fin:
            INFO_DICT = json.load(fin)


load_info_dict()


def save_info_dict():
    with open(INFO_FNAME, 'w') as fout:
        json.dump(INFO_DICT, fout)


def find_all_file(path, exts):
    result = []
    for name in os.listdir(path):
        if name.startswith('.'):
            # ignore the hidden directory
            continue
        fname = os.path.join(path, name)
        if os.path.isdir(fname):
            result.extend(find_all_file(fname, exts))
        else:
            ext = os.path.splitext(name)[1]
            if ext in exts:
                result.append(fname)
    return result


def get_file_hash(fname):
    return str(int(os.path.getmtime(fname)))


def file_is_changed(fname):
    global INFO_DICT
    fname = os.path.abspath(fname)
    cur_hash = get_file_hash(fname)
    return INFO_DICT.get(fname, None) != cur_hash


def update_file_hash(fname):
    global INFO_DICT
    fname = os.path.abspath(fname)
    cur_hash = get_file_hash(fname)
    INFO_DICT[fname] = cur_hash


def clang_format(fnames):
    for fname in fnames:
        if file_is_changed(fname):
            print('Format {}'.format(fname))
            script = 'clang-format -style="{BasedOnStyle: Google, Standard: Cpp11}" -i ' + fname
            os.system(script)
            update_file_hash(fname)


def autopep8(fnames):
    for fname in fnames:
        if file_is_changed(fname):
            print('Format {}'.format(fname))
            os.system('autopep8 --ignore E402 --in-place {}'.format(fname))
            update_file_hash(fname)


def filter_ignore_path(fnames, ignore_path):
    abs_ignore_path = os.path.abspath(ignore_path)

    def is_not_ignore_path(fname):
        abs_fname = os.path.abspath(fname)
        return not abs_fname.startswith(abs_ignore_path)

    return filter(is_not_ignore_path, fnames)


if __name__ == '__main__':
    ignore_path = './mobula/op/templates/'
    cpp_res = find_all_file('./', ['.cpp', '.h'])
    cpp_res = filter_ignore_path(cpp_res, ignore_path)
    clang_format(cpp_res)

    py_res = find_all_file('./', ['.py'])
    py_res = filter_ignore_path(py_res, ignore_path)
    autopep8(py_res)
    save_info_dict()

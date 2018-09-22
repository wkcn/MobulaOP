import os
from mobula.build_utils import build_context, file_changed, update_file_hash


def find_all_file(path, exts):
    result = []
    for name in os.listdir(path):
        fname = os.path.join(path, name)
        if os.path.isdir(fname):
            result.extend(find_all_file(fname, exts))
        else:
            ext = os.path.splitext(name)[1]
            if ext in exts:
                result.append(fname)
    return result


def clang_format(fnames):
    for fname in fnames:
        if file_changed(fname):
            print('Format {}'.format(fname))
            os.system('clang-format -style=google -i {}'.format(fname))
            update_file_hash(fname)


def autopep8(fnames):
    for fname in fnames:
        if file_changed(fname):
            print('Format {}'.format(fname))
            os.system('autopep8 --in-place {}'.format(fname))
            update_file_hash(fname)


with build_context():
    cpp_res = find_all_file('./', ['.cpp', '.h'])
    clang_format(cpp_res)

    py_res = find_all_file('./', ['.py'])
    autopep8(py_res)

import os
from mobula.build_utils import build_context, file_is_changed, update_file_hash, update_build_path


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


def clang_format(fnames):
    for fname in fnames:
        if file_is_changed(fname):
            print('Format {}'.format(fname))
            script = 'clang-format -style="{BasedOnStyle: Google, Standard: Cpp11}" -i ' + fname
            if os.system(script) == 0:
                update_file_hash(fname)


def autopep8(fnames):
    for fname in fnames:
        if file_is_changed(fname):
            print('Format {}'.format(fname))
            if os.system('autopep8 --ignore E402 --in-place {}'.format(fname)) == 0:
                update_file_hash(fname)


def filter_ignore_path(fnames, ignore_path):
    abs_ignore_path = os.path.abspath(ignore_path)

    def is_not_ignore_path(fname):
        abs_fname = os.path.abspath(fname)
        return not abs_fname.startswith(abs_ignore_path)

    return filter(is_not_ignore_path, fnames)


if __name__ == '__main__':
    update_build_path('./autoformat_code')
    ignore_path = './mobula/op/templates/'
    with build_context():
        cpp_res = find_all_file('./', ['.cpp', '.h'])
        cpp_res = filter_ignore_path(cpp_res, ignore_path)
        clang_format(cpp_res)

        py_res = find_all_file('./', ['.py'])
        py_res = filter_ignore_path(py_res, ignore_path)
        autopep8(py_res)

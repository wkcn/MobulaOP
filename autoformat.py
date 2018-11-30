import os
from mobula.build_utils import build_context, file_changed, update_file_hash, update_build_path


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
            script = 'clang-format -style="{BasedOnStyle: Google, Standard: Cpp11}" -i ' + fname
            os.system(script)
            update_file_hash(fname)


def autopep8(fnames):
    for fname in fnames:
        if file_changed(fname):
            print('Format {}'.format(fname))
            os.system('autopep8 --ignore E402 --in-place {}'.format(fname))
            update_file_hash(fname)


if __name__ == '__main__':
    update_build_path('./autoformat_code')
    with build_context():
        cpp_res = find_all_file('./', ['.cpp', '.h'])
        clang_format(cpp_res)

        py_res = find_all_file('./', ['.py'])
        autopep8(py_res)

"""Building Utils"""
__all__ = ["pass_argv", "get_include_file", "wildcard",
           "change_ext", "change_exts", "mkdir", "rmdir", "add_path",
           "file_changed", "file_is_latest",
           "run_command", "run_command_parallel", "command_exists",
           "config", "Flags", "INC_PATHS", "ENV_PATH",
           "OS_IS_WINDOWS", "OS_IS_LINUX", "build_context"]

from .config import config
import ast
import os
import threading
import platform
import re
from subprocess import Popen, PIPE
try:
    import Queue
except ImportError:
    import queue as Queue
if not hasattr(Queue.Queue, 'clear'):
    def _queue_clear(self):
        with self.mutex:
            self.queue.clear()
    setattr(Queue.Queue, 'clear', _queue_clear)

OS_NAME = platform.system()
OS_IS_WINDOWS = OS_NAME == 'Windows'
OS_IS_LINUX = OS_NAME in ['Linux', 'Darwin']
assert OS_IS_WINDOWS or OS_IS_LINUX,\
    Exception('Unsupported Operator System: {}'.format(OS_NAME))

INC_PATHS = ['./']

# Load Config File
ENV_PATH = os.path.dirname(__file__)
if not os.path.dirname(config.BUILD_PATH):
    config.BUILD_PATH = os.path.join(ENV_PATH, config.BUILD_PATH)


def pass_argv(argv):
    """Read Config from argv"""
    for p in argv:
        if p[0] == '-' and '=' in p:
            k, v = p[1:].split('=')
            k = k.strip()
            assert hasattr(config, k), KeyError(
                'Key `%s` not found in config' % k)
            setattr(config, k, ast.literal_eval(v))
            print('Set %s to %s' % (k, v))


def save_code_hash(obj, fname):
    with open(fname, 'w') as f:
        for k, v in obj.items():
            f.write('%s %s\n' % (k, v))


def load_code_hash(fname):
    data = dict()
    try:
        with open(fname, 'r') as f:
            for line in f:
                sp = line.split(' ')
                data[sp[0]] = sp[1].strip()
    except Exception:
        pass
    return data


def save_dependant(obj, fname):
    with open(fname, 'w') as f:
        for k, v in obj.items():
            if v:
                s = '{} {}\n'.format(k, ','.join(v))
                f.write(s)


def load_dependant(fname):
    data = dict()
    try:
        with open(fname, 'r') as f:
            for line in f:
                sp = line.strip().split(' ')
                data[sp[0]] = sp[1].split(',')
    except Exception:
        pass
    return data


def update_build_path(build_path):
    global code_hash, code_hash_filename, code_hash_updated
    global dependant, dependant_filename, dependant_updated

    os.makedirs(build_path, exist_ok=True)

    config.BUILD_PATH = build_path

    code_hash = dict()
    code_hash_filename = os.path.join(config.BUILD_PATH, 'code.hash')
    if os.path.exists(code_hash_filename):
        code_hash = load_code_hash(code_hash_filename)
    code_hash_updated = False

    dependant = dict()
    dependant_filename = os.path.join(config.BUILD_PATH, 'code.dependant')
    if os.path.exists(dependant_filename):
        dependant = load_dependant(dependant_filename)
    dependant_updated = False


update_build_path(os.path.join(ENV_PATH, config.BUILD_PATH))


def build_exit():
    if code_hash_updated:
        save_code_hash(code_hash, code_hash_filename)
    if dependant_updated:
        save_dependant(dependant, dependant_filename)


class build_context:
    def __enter__(self):
        pass

    def __exit__(self, *dummy):
        build_exit()


class Flags:
    def __init__(self, s=''):
        self.flags = s

    def add_definition(self, key, value):
        if isinstance(value, bool):
            value = int(value)
        self.flags += ' -D%s=%s' % (key, str(value))
        return self

    def add_string(self, s):
        self.flags += ' %s' % str(s)
        return self

    def __str__(self):
        return self.flags


INCLUDE_FILE_REG = re.compile(r'^\s*#include\s*(?:"|<)\s*(.*?)\s*(?:"|>)\s*')


def get_include_file(fname):
    res = []
    for line in open(fname):
        u = INCLUDE_FILE_REG.search(line)
        if u is not None:
            inc_fname = u.groups()[0]
            res.append(inc_fname)
    return res


def wildcard(path, ext):
    if isinstance(path, (list, tuple)):
        res = []
        for p in path:
            res.extend(wildcard(p, ext))
        return res
    res = []
    for name in os.listdir(path):
        e = os.path.splitext(name)[1]
        if e == '.' + ext:
            res.append(os.path.join(path, name))
    return res


def change_exts(lst, rules):
    res = []
    mappings = dict(rules)
    for name in lst:
        sp = os.path.splitext(name)
        if sp[1] and sp[1][0] == '.':
            ext = sp[1][1:]
            if ext in mappings:
                new_ext = mappings[ext]
                name = sp[0] + '.' + new_ext
        res.append(name)
    return res


def change_ext(lst, origin, target):
    return change_exts(lst, [(origin, target)])


def run_command(command):
    print(command)
    return os.system(command)


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        print('mkdir -p %s' % dir_name)
        os.makedirs(dir_name, exist_ok=True)


if OS_IS_LINUX:
    rmdir_command = 'rm -rf'
elif OS_IS_WINDOWS:
    rmdir_command = 'rd /s /q'


def rmdir(dir_name):
    # we use shell command to remove the non-empty or empry directory
    if os.path.exists(dir_name):
        command = '%s %s' % (rmdir_command, dir_name)
        run_command(command)


def get_file_hash(fname):
    return str(int(os.path.getmtime(fname)))


def file_changed(fname):
    fname = os.path.abspath(fname)
    global code_hash_updated
    new_hash = get_file_hash(fname)
    if fname not in code_hash or new_hash != code_hash[fname]:
        code_hash_updated = True
        code_hash[fname] = new_hash
        return True
    return False


def update_file_hash(fname):
    fname = os.path.abspath(fname)
    global code_hash_updated
    new_hash = get_file_hash(fname)
    if fname not in code_hash or new_hash != code_hash[fname]:
        code_hash_updated = True
        code_hash[fname] = new_hash


def find_include(inc):
    for path in INC_PATHS:
        fname = os.path.relpath(os.path.join(
            ENV_PATH, path, inc), start=ENV_PATH)
        if os.path.exists(fname):
            return fname
    return None


def is_c_file(fname):
    return os.path.splitext(fname)[-1] not in ['.cpp', '.c', '.cu']


def update_dependant(fname):
    if is_c_file(fname):
        return
    fname = os.path.abspath(fname)
    global dependant_updated
    dependant_updated = True
    inc_files = get_include_file(fname)
    res = []
    for inc in inc_files:
        inc_fname = find_include(inc)
        if inc_fname is not None:
            inc_fname = os.path.abspath(inc_fname)
            res.append(inc_fname)
    dependant[fname] = res


def dependant_changed(fname):
    if is_c_file(fname):
        return False
    fname = os.path.abspath(fname)
    if fname not in dependant:
        update_dependant(fname)
    includes = dependant[fname]
    changed = False
    for inc in includes:
        inc_fname = find_include(inc)
        if inc_fname is not None:
            abs_inc_fname = os.path.abspath(inc_fname)
            if not file_is_latest(abs_inc_fname):
                changed = True
    return changed


FILE_CHECK_LIST = dict()


def file_is_latest(source):
    source = os.path.abspath(source)
    if source in FILE_CHECK_LIST:
        t = FILE_CHECK_LIST[source]
        assert t is not None, RuntimeError(
            'Error: Cycle Reference {}'.format(source))
        return t
    FILE_CHECK_LIST[source] = None
    latest = True
    if file_changed(source):
        latest = False
        update_dependant(source)
    if dependant_changed(source):
        latest = False
    FILE_CHECK_LIST[source] = latest
    return latest


def run_command_parallel(commands, allow_error=False):
    command_queue = Queue.Queue()
    info_queue = Queue.Queue()
    for c in commands:
        command_queue.put(c)
    max_worker_num = min(config.MAX_BUILDING_WORKER_NUM, len(commands))
    for _ in range(max_worker_num):
        command_queue.put(None)

    def worker(command_queue, info_queue):
        while not command_queue.empty():
            e = command_queue.get()
            if e is None:
                break
            rtn = run_command(e)
            if rtn != 0:
                # Error
                command_queue.clear()
                info_queue.put(Exception('Error, terminated :-('))
    workers = [threading.Thread(target=worker, args=(command_queue, info_queue))
               for _ in range(max_worker_num)]
    for w in workers:
        w.daemon = True
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    while not info_queue.empty():
        info = info_queue.get()
        if isinstance(info, Exception) and not allow_error:
            raise RuntimeError(info)


def add_path(path, files):
    return list(map(lambda x: os.path.join(path, x), files))


def command_exists(command):
    try:
        Popen([command], stdout=PIPE, stderr=PIPE, stdin=PIPE)
    except Exception:
        return False
    return True

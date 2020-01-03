import sys
import os

if sys.version_info[0] < 3:
    old_os_makedirs = os.makedirs

    def makedirs(name, mode=511, exist_ok=False):
        '''makedirs(name [, mode=0o777][, exist_ok=False])

Super-mkdir; create a leaf directory and all intermediate ones.  Works like
mkdir, except that any intermediate path segment (not just the rightmost)
will be created if it does not exist. If the target directory already
exists, raise an OSError if exist_ok is False. Otherwise no exception is
raised.  This is recursive.'''
        try:
            old_os_makedirs(name, mode)
        except OSError:
            if not exist_ok:
                raise

    os.makedirs = makedirs

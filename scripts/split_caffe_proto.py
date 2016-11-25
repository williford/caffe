import mmap
import re
import os
import errno

# a regex to match the parameter definitions in caffe.proto
r = re.compile(r'(?://.*\n)*' +
               r'message ([^ ]*) \{\n' +
               r'(?: .*\n|' +
               r'\n)*\}')

# create directory to put caffe.proto fragments
try:
    os.mkdir('../docs/tutorial/proto/')
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

with open('../src/caffe/proto/caffe.proto', 'r') as fin:

    for m in r.finditer(fin.read(), re.MULTILINE):
        fn = '../docs/tutorial/proto/%s.txt' % m.group(1)
        print m.group(1)
        print m.group(0)
        with open(fn, 'w') as fout:
            fout.write(m.group(0))
        print '------------------------------------------'

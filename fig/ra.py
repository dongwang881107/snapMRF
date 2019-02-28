# This file is part of the RA package (http://github.com/davidssmith/ra).
#
# The MIT License (MIT)
#
# Copyright (c) 2015-2017 David Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import struct
import numpy as np

FLAG_BIG_ENDIAN = 0x01
MAGIC_NUMBER = 8746397786917265778
dtype_kind_to_enum = {'i':1,'u':2,'f':3,'c':4}
dtype_enum_to_name = {0:'user',1:'int',2:'uint',3:'float',4:'complex'}

def read(filename):
    f = open(filename,'rb')
    h = getheader(f)
    h['dims'] = h['dims'][::-1]
    if h['eltype'] == 0:
        print('Unable to convert user data. Returning raw byte string.')
        data = f.read(h['size'])
    else:
        d = '%s%d' % (dtype_enum_to_name[h['eltype']], h['elbyte']*8)
        data = np.fromstring(f.read(h['size']), dtype=np.dtype(d))
        data = data.reshape(h['dims']).transpose()
    f.close()
    return data


def getheader(f):
    filemagic = f.read(8)
    h = dict()
    h['flags'] = struct.unpack('<Q',f.read(8))[0]
    h['eltype'] = struct.unpack('<Q',f.read(8))[0]
    h['elbyte'] = struct.unpack('<Q',f.read(8))[0]
    h['size'] = struct.unpack('<Q',f.read(8))[0]
    h['ndims'] = struct.unpack('<Q',f.read(8))[0]
    h['dims'] = []
    for d in range(h['ndims']):
        h['dims'].append(struct.unpack('<Q',f.read(8))[0])
    return h


def query(filename):
  q = "---\nname: %s\n" % filename
  fd = open(filename,'r')
  h = getheader(fd)
  fd.close()
  if h['flags'] & FLAG_BIG_ENDIAN:
    endian = 'big'
  else:
    endian = 'little'
  assert endian == 'little'  # big not implemented yet
  q += 'endian: %s\n' % endian
  q += 'type: %s%d\n' % (dtype_enum_to_name[h['eltype']], h['elbyte']*8)
  q += 'size: %d\n' % h['size']
  q += 'dimension: %d\n' % h['ndims']
  q += 'shape:\n'
  for d in h['dims']:
     q += '  - %d\n' % d
  q += '...'
  return q


def write(data, filename):
    flags = 0
    if data.dtype.str[0] == '>':
        flags |= FLAG_BIG_ENDIAN
    try:
        eltype = dtype_kind_to_enum[data.dtype.kind]
    except KeyError:
        eltype = 0
    elbyte = data.dtype.itemsize
    size = data.size*elbyte
    ndims = len(data.shape)
    dims = data.shape
    dims = np.array([_ for _ in data.shape]).astype('uint64')
    f = open(filename,'wb')
    f.write(struct.pack('<Q', MAGIC_NUMBER))
    f.write(struct.pack('<Q', flags))
    f.write(struct.pack('<Q', eltype))
    f.write(struct.pack('<Q', elbyte))
    f.write(struct.pack('<Q', size))
    f.write(struct.pack('<Q', ndims))
    f.write(dims.tobytes())
    f.write(data.transpose().tobytes())
    f.close()

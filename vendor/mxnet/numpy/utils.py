# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Util functions for the numpy module."""



import numpy as np

__all__ = ['float16', 'float32', 'float64', 'uint8', 'int32', 'int8', 'int64',
           'bool', 'bool_', 'pi', 'inf', 'nan', 'PZERO', 'NZERO', 'newaxis', 'finfo',
           'e', 'NINF', 'PINF', 'NAN', 'NaN',
           '_STR_2_DTYPE_']

float16 = np.float16
float32 = np.float32
float64 = np.float64
uint8 = np.uint8
int32 = np.int32
int8 = np.int8
int64 = np.int64
bool_ = bool
bool = bool

pi = np.pi
inf = np.inf
nan = np.nan
PZERO = 0.0
NZERO = -0.0
NINF = -np.inf
PINF = np.inf
e = np.e
NAN = nan
NaN = np.nan

newaxis = None
finfo = np.finfo

_STR_2_DTYPE_ = {'float16': float16, 'float32': float32, 'float64':float64, 'float': float64,
                 'uint8': uint8, 'int8': int8, 'int32': int32, 'int64': int64, 'int': int64,
                 'bool': bool, 'bool_': bool_, 'None': None}

_ONP_OP_MODULES = [np, np.linalg, np.random, np.fft]


def _get_np_op(name):
    """Get official NumPy operator with `name`. If not found, raise ValueError."""
    for mod in _ONP_OP_MODULES:
        op = getattr(mod, name, None)
        if op is not None:
            return op
    raise ValueError('Operator `{}` is not supported by `mxnet.numpy`.'.format(name))

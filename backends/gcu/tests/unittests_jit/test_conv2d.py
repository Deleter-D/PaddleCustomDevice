# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from api_base import ApiBase
import paddle
import pytest
import numpy as np


@pytest.mark.conv2d
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_conv2d():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=["data", "kernel"],
        feed_shapes=[[1, 2, 8, 8], [4, 2, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 2, 8, 8)).astype("float32")
    kernel = np.random.uniform(-1, 1, (4, 2, 3, 3)).astype("float32")
    test.run(feed=[data, kernel], bias=None, stride=2, padding=1)


@pytest.mark.conv2d
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_conv2d_1():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=["data", "kernel"],
        feed_shapes=[[1, 3, 704, 1280], [8, 3, 3, 3]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 3, 704, 1280)).astype("float32")
    kernel = np.random.uniform(-1, 1, (8, 3, 3, 3)).astype("float32")
    test.run(feed=[data, kernel], bias=None, stride=2, padding=1)


@pytest.mark.conv2d
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_conv2d_2():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=["data", "kernel"],
        feed_shapes=[[1, 8, 352, 640], [8, 8, 1, 1]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (1, 8, 352, 640)).astype("float32")
    kernel = np.random.uniform(-1, 1, (8, 8, 1, 1)).astype("float32")
    test.run(feed=[data, kernel], bias=None, stride=1, padding=0)


@pytest.mark.conv2d
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_conv2d_3():
    test = ApiBase(
        func=paddle.nn.functional.conv2d,
        feed_names=["data", "kernel"],
        feed_shapes=[[16, 240, 1, 1], [60, 240, 1, 1]],
    )
    np.random.seed(1)
    data = np.random.uniform(-1, 1, (16, 240, 1, 1)).astype("float32")
    kernel = np.random.uniform(-1, 1, (60, 240, 1, 1)).astype("float32")
    test.run(feed=[data, kernel], bias=None, stride=1, padding=0)

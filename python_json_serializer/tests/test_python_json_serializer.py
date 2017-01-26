# -*- coding: utf-8 -*-
import unittest
import chainer
import io
import python_json_serializer
import json
import numpy


class SampleChain(chainer.Chain):
    def __init__(self):
        super().__init__(
            f1=chainer.links.Linear(in_size=3, out_size=2),
            f2=chainer.links.Linear(in_size=2, out_size=1)
        )

    def __call__(self, x):
        return self.f2(self.f1(x))


class JsonSerializerTest(unittest.TestCase):
    def test_serialize(self):
        obj = SampleChain()
        with io.StringIO() as f:
            python_json_serializer.save_json_file(f, obj)
            f.flush()
            dict = json.loads(f.getvalue())
            self.assertTrue("f1/W" in dict)

    def test_serialize_deserialize(self):
        obj = SampleChain()
        obj2 = SampleChain()
        input_variable = chainer.Variable(numpy.arange(3, dtype=numpy.float32).reshape(1, 1, 1, 3))
        self.assertRaises(
            AssertionError,
            numpy.testing.assert_array_equal,
            obj(input_variable).data,
            obj2(input_variable).data
        )
        with io.StringIO() as f_write:
            python_json_serializer.save_json_file(f_write, obj)
            f_write.flush()
            with io.StringIO(f_write.getvalue()) as f_read:
                python_json_serializer.load_json_file(f_read, obj2)
        numpy.testing.assert_array_equal(
            obj(input_variable).data,
            obj2(input_variable).data
        )

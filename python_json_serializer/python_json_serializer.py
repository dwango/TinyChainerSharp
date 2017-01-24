# -*- coding: utf-8 -*-
import chainer
import json

import numpy
from chainer import cuda


def save_json(filename, obj):
    with open(filename, 'w') as f:
        save_json_file(f, obj)


def save_json_file(filobj, obj):
    s = chainer.serializers.DictionarySerializer()
    s.save(obj)
    print(s.target)
    for key in s.target.keys():
        s.target[key] = s.target[key].tolist()
    json.dump(s.target, filobj)


class JSONDeserializer(chainer.serializer.Deserializer):
    def __init__(self, dict, path=''):
        self.dict = dict
        self.path = path

    def __getitem__(self, key):
        key = key.strip('/')
        return JSONDeserializer(self.dict, self.path + key + '/')

    def __call__(self, key, value):
        dataset = self.dict[self.path + key]
        if value is None:
            return dataset
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(numpy.asarray(dataset))
        return value


def load_json_file(fileobj, obj):
    d = JSONDeserializer(json.load(fileobj))
    d.load(obj)

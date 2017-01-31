using System.Collections.Generic;
using System.Diagnostics;

namespace chainer.serializers
{
    public abstract class Serializer
    {
        public abstract Serializer Traverse(string key);
        public abstract void Communicate(string key, Variable value);
    }

    public class DictionarySerializer : Serializer
    {
        public Dictionary<string, Variable> Target;
        private readonly string _path;

        public DictionarySerializer(Dictionary<string, Variable> target = null, string path = "")
        {
            Target = target ?? new Dictionary<string, Variable>();
            _path = path;
        }

        public override Serializer Traverse(string key)
        {
            return new DictionarySerializer(Target, _path + key + "/");
        }

        public override void Communicate(string key, Variable value)
        {
            Debug.Assert(!Target.ContainsKey(key));
            Target[_path + key] = value;
        }
    }


    public class DictionaryDeserializer : Serializer
    {
        private Dictionary<string, Variable> _source;
        private readonly string _path;

        public DictionaryDeserializer(Dictionary<string, Variable> source, string path = "")
        {
            _source = source;
            _path = path;
        }

        public override Serializer Traverse(string key)
        {
            return new DictionarySerializer(_source, _path + key + "/");
        }

        public override void Communicate(string key, Variable value)
        {
            var fullpath = _path + key;
            if (_source.ContainsKey(fullpath))
            {
                Debug.Assert(value.Value.RowCount == _source[fullpath].Value.RowCount);
                Debug.Assert(value.Value.ColumnCount == _source[fullpath].Value.ColumnCount);
                value.Value = _source[fullpath].Value.Clone();
            }
        }
    }
}
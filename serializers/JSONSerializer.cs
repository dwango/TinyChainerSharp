using System.Collections.Generic;
using System.IO;
using System.Net;
using SimpleJSON;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.serializers
{
    public class JsonSerializer : DictionarySerializer
    {
        public JsonSerializer(Dictionary<string, Variable> target = null, string path = "") : base(target, path)
        {
        }

        private static JSONNode ParseVariable(Variable variable)
        {
            var json = JSON.Parse("[]");
            foreach (var row in variable.Value.ToRowArrays())
            {
                var rowjson = JSON.Parse("[]");
                foreach (var value in row)
                {
                    rowjson[-1] = new JSONData(value);
                }
                json[-1] = rowjson;
            }
            return json;
        }

        public string Fetch()
        {
            var json = JSON.Parse("{}");
            foreach (var kv in Target)
            {
                json[kv.Key] = ParseVariable(kv.Value);
            }
            return json.ToString();
        }
    }

    public class JsonDeserializer : Serializer
    {
        private readonly JSONNode _source;
        private readonly string _path;

        public JsonDeserializer(JSONNode source, string path = "")
        {
            _source = source;
            _path = path;
        }

        public JsonDeserializer(string jsonfilepath, string path = "") : this(JSONNode.LoadFromFile(jsonfilepath), path)
        {
        }

        public override Serializer Traverse(string key)
        {
            return new JsonDeserializer(_source, _path + key + "/");
        }

        public override void Communicate(string key, Variable value)
        {
            var fullpath = _path + key;

            // check variable depth because it can only handle Matrix
            // TODO(ogaki): handle tensor whose dimention > 2
            var matrixNode = _source[fullpath];
            for (var currentNode = _source[fullpath]; currentNode[0].AsArray != null; currentNode = currentNode[0])
            {
                matrixNode = currentNode;
            }

            if (matrixNode[0].AsArray == null) // if vector
            {
                var tmp = JSON.Parse("[]");
                tmp[-1] = matrixNode;
                matrixNode = tmp;
            }

            var matrix = new float[matrixNode.AsArray.Count, matrixNode[0].AsArray.Count];
            for (int x = 0; x < matrix.GetLength(0); x++)
            {
                for (int y = 0; y < matrix.GetLength(1); y++)
                {
                    matrix[x, y] = matrixNode[x][y].AsFloat;
                }
            }
            value.Value = Matrix<float>.Build.DenseOfArray(matrix);
        }
    }
}
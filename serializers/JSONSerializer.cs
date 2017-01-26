using System.Collections.Generic;
using SimpleJSON;

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
            json["a"][-1][-1][-1] = "1";
            foreach (var kv in Target)
            {
                json[kv.Key] = ParseVariable(kv.Value);
            }
            return json;
        }
    }
}
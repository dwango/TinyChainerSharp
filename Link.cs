using System;
using System.Collections.Generic;
using System.Linq;

namespace chainer
{
    public class Link
    {
        public Dictionary<string, Variable> _Params = new Dictionary<string, Variable>();

        public Link(Dictionary<string, Variable> @params)
        {
            _Params = @params;
        }

        protected Link()
        {
        }

        public virtual IEnumerable<Variable> GetParams()
        {
            return _Params.Values.AsEnumerable();
        }

        public void ClearGrads()
        {
            foreach (var param in GetParams())
            {
                param.ClearGrad();
            }
        }

        public virtual Variable Forward(Variable x)
        {
            throw new NotImplementedException();
        }

        public virtual void Serialize(serializers.Serializer serializer)
        {
            foreach (var kv in _Params)
            {
                serializer.Communicate(kv.Key, kv.Value);
            }
        }
    }
}
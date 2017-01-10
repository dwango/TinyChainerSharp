using System.Collections.Generic;
using System.Linq;

namespace chainer
{
    public class Link
    {
        protected Dictionary<string, Variable> Params = new Dictionary<string, Variable>();

        public Link(Dictionary<string, Variable> @params)
        {
            Params = @params;
        }

        protected Link()
        {
        }

        public virtual IEnumerable<Variable> GetParams()
        {
            return Params.Values.AsEnumerable();
        }

        public void ClearGrads()
        {
            foreach (var param in GetParams())
            {
                param.ClearGrad();
            }
        }
    }
}
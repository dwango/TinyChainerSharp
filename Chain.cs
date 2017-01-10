using System.Collections.Generic;

namespace chainer
{
    public class Chain
    {
        public Dictionary<string, Variable> Params;

        public Chain(Dictionary<string, Variable> @params)
        {
            Params = @params;
        }

        public void ClearGrads()
        {
            foreach (var kv in Params)
            {
                kv.Value.ClearGrad();
            }
        }
    }
}
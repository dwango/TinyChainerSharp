using System.Collections.Generic;
using System.Linq;

namespace chainer
{
    public class Chain : Link
    {
        protected Dictionary<string, Link> Children = new Dictionary<string, Link>();

        public Chain(Dictionary<string, Link> @children)
        {
            Children = @children;
        }


        public override IEnumerable<Variable> GetParams()
        {
            var selfParams = Params.Values.AsEnumerable();
            var childrenParams = Children.Values.SelectMany(child => child.GetParams());
            return selfParams.Concat(childrenParams);
        }
    }
}
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.helper.models
{
    public class LogicalOperationChain : Chain
    {
        public LogicalOperationChain(
        ) : base(new Dictionary<string, Link>()
        {
            {"fc1", new links.Linear(2, 6)},
            {"fc2", new links.Linear(6, 1)}
        })
        {
            // seed固定
            Children["fc1"]._Params["W"].Value = Matrix<float>.Build.Random(6, 2, seed: 0);
            Children["fc1"]._Params["b"].Value = Matrix<float>.Build.Random(1, 6, seed: 0);
            Children["fc2"]._Params["W"].Value = Matrix<float>.Build.Random(1, 6, seed: 1);
            Children["fc2"]._Params["b"].Value = Matrix<float>.Build.Random(1, 1, seed: 1);
        }

        public override Variable Forward(Variable x)
        {
            var h = x;
            h = functions.Sigmoid.ForwardStatic(Children["fc1"].Forward(h));
            h = Children["fc2"].Forward(h);
            return h;
        }
    }
}
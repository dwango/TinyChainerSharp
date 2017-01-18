using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;

namespace chainer.helper.models
{
    public class VerySmallChain : Chain
    {
        public Link fc;

        public VerySmallChain() : base(new Dictionary<string, Link>()
            {
                {"fc", new links.Linear(3, 1)}
            }
        )
        {
            Children["fc"]._Params["W"].Value = Matrix<float>.Build.DenseOfArray(new float[,] {{-1, 0, 1}});
            Children["fc"]._Params["b"].Value = Matrix<float>.Build.DenseOfArray(new float[,] {{1}});
            fc = Children["fc"];
        }

        public override Variable Forward(Variable x)
        {
            return fc.Forward(x);
        }
    }
}
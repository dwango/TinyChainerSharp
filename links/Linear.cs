using MathNet.Numerics.LinearAlgebra;

namespace chainer.links
{
    public class Linear: Link
    {
        public Linear(int inSize, int outSize)
        {
            _Params["W"] = new Variable(Matrix<float>.Build.Random(outSize, inSize));
            _Params["b"] = new Variable(Matrix<float>.Build.Random(1, outSize));
        }

        public override Variable Forward(Variable x)
        {
            return functions.Linear.ForwardStatic(x, _Params["W"], _Params["b"]);
        }
    }
}

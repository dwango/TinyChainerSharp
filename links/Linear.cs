using MathNet.Numerics.LinearAlgebra.Single;

namespace chainer.links
{
    public class Linear
    {
        private Variable W;
        private Variable b;

        public Linear()
        {
            b = new Variable(DenseMatrix.OfArray(new float[,] {{1, 1, 1}}).Transpose());
            W = new Variable(DenseMatrix.OfDiagonalArray(new float[] {1, 1, 1}));
        }

        public Variable Forward(Variable x)
        {
            return new Variable(W.Value * x.Value + b.Value);
        }

//        public Variable Backward(Variable x, Variable gy)
//        {
//            return new Variable(W * x + b);
//        }
    }
}
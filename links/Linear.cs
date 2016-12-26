using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace chainer.links
{
    public class Linear
    {
        private Matrix<float> W;
        private Matrix<float> b;

        public Linear()
        {
            b = DenseMatrix.OfArray(new float[,]{{1, 1, 1}}).Transpose();
            W = DenseMatrix.OfDiagonalArray(new float[] {1, 1, 1});
        }

        public Matrix<float> Forward(Matrix<float> x)
        {
            return W * x + b;
        }
    }
}

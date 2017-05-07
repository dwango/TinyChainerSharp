using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer.functions
{
    public class MultiplyTest
    {
        [Test]
        public void Forwardがただしい()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}));
            var y = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{4, -1, 0.1f}}));
            var result = Multiply.ForwardStatic(
                x, y
            );

            chainer.Helper.AssertMatrixAlmostEqual(
                result.Value,
                Matrix<float>.Build.DenseOfArray(new float[,] {{4, -2, 0.3f}})
            );
        }
    }
}
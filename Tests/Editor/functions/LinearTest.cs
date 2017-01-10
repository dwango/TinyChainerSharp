using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer.functions
{
    public class LinearTest
    {
        [Test]
        public void Fowardの式が正しい()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}));
            var W = new Variable(Matrix<float>.Build.DenseOfArray(new float[,]
            {
                {3, 4, 3},
                {1, 2, 3},
                {5, 4, 1},
            }));
            var b = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{4, 2, 1}}));
            var testOutput = Linear.ForwardStatic(x, W, b);
            var expected = Matrix<float>.Build.DenseOfArray(new float[,] {{24, 16, 17}}); // by chainer

            chainer.Helper.MatrixAlmostEqual(testOutput.Value, expected);
        }

        [Test]
        public void Backwardの式が正しい()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}));
            var W = new Variable(Matrix<float>.Build.DenseOfArray(new float[,]
            {
                {3, 4, 3},
                {1, 2, 3},
                {5, 4, 1},
            }));
            var b = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{4, 2, 1}}));
            var target = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{0.5f, 0.5f, 0.5f}}));
            var loss = MeanSquaredError.ForwardStatic(
                Linear.ForwardStatic(x, W, b),
                target
            );
            loss.Backward();
            chainer.Helper.MatrixAlmostEqual(
                x.Grad,
                Matrix<float>.Build.DenseOfArray(new float[,] {{112.33333588f,  127.33333588f,   89f}})
            );
        }
    }
}
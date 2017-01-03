using MathNet.Numerics.LinearAlgebra.Single;
using NUnit.Framework;

namespace chainer.functions
{
    public class SigmoidTest
    {
        [Test]
        public void Fowardの式が正しい()
        {
            var testInput = new Variable(DenseMatrix.OfArray(new float[,] {{-100, -1, 0.5f, 1, 100}}));
            var testOutput = Sigmoid.ForwardStatic(testInput);
            var expected = DenseMatrix.OfArray(new float[,] {{0f, 0.26f, 0.62f, 0.73f, 1f}}); // by chainer

            chainer.Helper.MatrixAlmostEqual(testOutput.Value, expected);
        }

        [Test]
        public void Backardの式が正しい()
        {
            var testInput = new Variable(DenseMatrix.OfArray(new float[,] {{1, 2, 3}}));
            var target = new Variable(DenseMatrix.OfArray(new float[,] {{0.5f, 0.5f, 0.5f}}));
            var loss = MeanSquaredError.ForwardStatic(
                Sigmoid.ForwardStatic(testInput),
                target
            );
            loss.Backward();
            chainer.Helper.MatrixAlmostEqual(
                testInput.Grad,
                DenseMatrix.OfArray(new float[,] {{0.03028592f, 0.02665417f, 0.01363052f}}) // by chainer.py
            );
        }
    }
}
using MathNet.Numerics.LinearAlgebra.Single;
using NUnit.Framework;

namespace chainer.functions
{
    public class ReLUTest
    {
        [Test]
        public void Fowardの式が正しい()
        {
            var testInput = new Variable(DenseMatrix.OfArray(new float[,] {{-100, -1, 0.5f, 1, 100}}));
            var testOutput = ReLU.ForwardStatic(testInput);
            var expected = DenseMatrix.OfArray(new float[,] {{0, 0, 0.5f, 1f, 100f}}); // by chainer

            chainer.Helper.AssertMatrixAlmostEqual(testOutput.Value, expected);
        }

        [Test]
        public void Backardの式が正しい()
        {
            var testInput = new Variable(DenseMatrix.OfArray(new float[,] {{1, 2, 3}}));
            var target = new Variable(DenseMatrix.OfArray(new float[,] {{0.5f, 0.5f, 0.5f}}));
            var loss = MeanSquaredError.ForwardStatic(
                ReLU.ForwardStatic(testInput),
                target
            );
            loss.Backward();
            chainer.Helper.AssertMatrixAlmostEqual(
                DenseMatrix.OfArray(new float[,] {{0.33333334f, 1f, 1.66666675f}}), // by chainer.py
                testInput.Grad
            );
        }
    }
}
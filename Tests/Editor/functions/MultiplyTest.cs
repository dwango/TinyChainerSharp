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

            Helper.AssertMatrixAlmostEqual(
                result.Value,
                Matrix<float>.Build.DenseOfArray(new float[,] {{4, -2, 0.3f}}) // chainer-pythonn
            );
        }

        [Test]
        public void Backwardがただしい()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}));
            var y = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{4, -1, 0.1f}}));
            var result = Multiply.ForwardStatic(
                x, y
            );
            var target = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{0.5f, 0.5f, 0.5f}}));
            var loss = MeanSquaredError.ForwardStatic(
                result,
                target
            );
            UnityEngine.Debug.Log(loss.Value);
            loss.Backward();
            UnityEngine.Debug.Log(result.Grad);
            Helper.AssertMatrixAlmostEqual(
                x.Grad,
                Matrix<float>.Build.DenseOfArray(new float[,]
                    {{9.33333397f, 1.66666675f, -0.01333333f}}) // chainer-python
            );
        }
    }
}
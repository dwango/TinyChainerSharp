using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer.functions
{
    public class MeanSquaredErrorTest
    {
        [Test]
        public void Forwardできる()
        {
            var func = new MeanSquaredError();
            var loss = func.Forward(
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose()),
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            );
            Assert.AreEqual(
                loss.Value,
                Matrix<float>.Build.DenseOfArray(new float[,] {{5}})
            );
        }

        [Test]
        public void ForwardBackwardで形が変わらない()
        {
            var func = new MeanSquaredError();
            var x0 = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());

            var loss = func.Forward(
                x0,
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            );
            var gxs = func.Backward(loss.Value).ToList();
            Assert.AreEqual(gxs[0].ColumnCount, x0.Value.ColumnCount);
            Assert.AreEqual(gxs[0].RowCount, x0.Value.RowCount);
        }

        [Test]
        public void 簡単なoptimizeしてみる()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
            var constant = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
            var target = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose());

            var lr = 0.1f;

            var convergence = false;
            for (int i = 0; i < 100; i++)
            {
                var loss = MeanSquaredError.ForwardStatic(
                    Add.ForwardStatic(x, constant),
                    target
                );

                x.Grad = null;
                loss.Backward();
                x.Value -= x.Grad * lr;
                if (loss.Value[0, 0] < 0.1f)
                {
                    convergence = true;
                    break;
                }
            }
            Assert.IsTrue(convergence);
        }
    }
}
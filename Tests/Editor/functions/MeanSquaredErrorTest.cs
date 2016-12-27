using System.Collections.Generic;
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
            var loss = func.Forward(new List<Variable>()
            {
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose()),
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            });
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

            var loss = func.Forward(new List<Variable>()
            {
                x0,
                new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose())
            });
            var gxs = func.Backward(loss.Value).ToList();
            Assert.AreEqual(gxs[0].ColumnCount, x0.Value.ColumnCount);
            Assert.AreEqual(gxs[0].RowCount, x0.Value.RowCount);
        }
    }
}
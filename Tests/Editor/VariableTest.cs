using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer
{
    public class VariableTest
    {
        [Test]
        public void backwardが計算グラフ上を伝搬する()
        {
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
            var constant = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 1, 1}}).Transpose());
            var target = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3}}).Transpose());
            var loss = functions.MeanSquaredError.ForwardStatic(
                x + constant,
                target
            );
            Assert.IsNull(x.Grad);
            loss.Backward();
            Assert.IsNotNull(x.Grad);
        }

    }
}
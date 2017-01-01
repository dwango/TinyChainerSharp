using MathNet.Numerics.LinearAlgebra.Single;
using NUnit.Framework;

namespace chainer.functions
{
    public class AddTest
    {
        [Test]
        public void Forwardできる()
        {
            var result = Add.ForwardStatic(
                new Variable(DenseMatrix.OfArray(new float[,] {{1, 1, 1}})),
                new Variable(DenseMatrix.OfArray(new float[,] {{1, 2, 1}}))
            );
            Assert.AreEqual(result.Value, DenseMatrix.OfArray(new float[,] {{2, 3, 2}}));
        }
    }
}

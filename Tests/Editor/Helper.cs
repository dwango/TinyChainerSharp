using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer
{
    public static class Helper
    {
        public static void MatrixAlmostEqual(Matrix<float> m1, Matrix<float> m2, float delta = 0.01f)
        {
            for (var i = 0; i < m1.ColumnCount; i++)
            {
                for (var j = 0; j < m1.RowCount; j++)
                {
                    Assert.AreEqual(
                        expected: m1[j, i],
                        actual: m2[j, i],
                        delta: delta
                    );
                }
            }
        }

        public static void MatrixNotAlmostEqual(Matrix<float> m1, Matrix<float> m2, float delta = 0.01f)
        {
            Assert.Throws<AssertionException>(
                () => { MatrixAlmostEqual(m1, m2, delta); }
            );
        }
    }
}
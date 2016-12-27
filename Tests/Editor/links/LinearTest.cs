﻿using MathNet.Numerics.LinearAlgebra.Single;
using NUnit.Framework;

namespace chainer.links
{
    public class LinearTest
    {
        [Test]
        public void Forwardできる()
        {
            var L = new Linear();
            var x = new Variable(DenseMatrix.OfArray(new float[,] {{1, 1, 1}}).Transpose());
            var y = L.Forward(x);
            Assert.AreEqual(y.Value[1, 0], 2.0f);
        }
    }
}
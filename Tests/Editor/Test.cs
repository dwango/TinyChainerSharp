using UnityEngine;
using UnityEditor;
using NUnit.Framework;
using MathNet.Numerics.LinearAlgebra.Double;

public class Test
{
    [Test]
    public void EditorTest()
    {
        Assert.IsTrue(true);
    }

    [Test]
    public void ベクトル演算()
    {
        var ones = new DenseVector(new double[] {1, 1, 1});
        var twos = ones * 2;
        Assert.AreEqual(twos[2], 2.0d);
    }
}
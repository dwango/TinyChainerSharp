using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;
using UnityEngine;

namespace chainer.serializers
{
    public class JSONSerializerTest
    {
        [Test]
        public void JSONへのシリアライズ()
        {
            var link = new helper.models.VerySmallChain();
            var serializer = new JsonSerializer();
            link.Serialize(serializer);
            Assert.True(serializer.Fetch().Contains("fc/W"));
        }

        [Test]
        public void JSONでのSaveLoad()
        {
            var link = new helper.models.VerySmallChain();
            var serializer = new JsonSerializer();
            link.Serialize(serializer);

            var link2 = new helper.models.VerySmallChain();
            link2.fc._Params["W"].Value[0, 0] = -100;

            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3,}}));
            Helper.AssertMatrixNotAlmostEqual(
                link.Forward(x).Value,
                link2.Forward(x).Value
            );

            var deserializer = new JsonDeserializer(SimpleJSON.JSON.Parse(serializer.Fetch()));
            link2.Serialize(deserializer);
            Helper.AssertMatrixAlmostEqual(
                link.Forward(x).Value,
                link2.Forward(x).Value
            );
        }

        [Test]
        public void Pythonでsaveしたパラメタを復元できる()
        {
            var pythonline =
                "{\"fc/W\": [[0.3128162920475006, 0.12374541908502579, -0.10456687211990356]], \"fc/b\": [0.0]}";
            var link = new helper.models.VerySmallChain();
            var deserializer = new JsonDeserializer(SimpleJSON.JSON.Parse(pythonline));
            var expected = Matrix<float>.Build.DenseOfArray(new float[,] {{0.24660653f}}); // calculated with python
            var x = new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{1, 2, 3,}}));

            Helper.AssertMatrixNotAlmostEqual(
                link.Forward(x).Value,
                expected
            );
            link.Serialize(deserializer);
            Helper.AssertMatrixAlmostEqual(
                link.Forward(x).Value,
                expected
            );
        }
    }
}
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

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
            UnityEngine.Debug.Log(serializer.Fetch());

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
            
        }
    }
}
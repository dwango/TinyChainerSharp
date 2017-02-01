using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using NUnit.Framework;

namespace chainer.serializers
{
    public class SerializerTest
    {
        [Test]
        public void Dictへのシリアライズ()
        {
            var link = new helper.models.VerySmallChain();
            var serializer = new DictionarySerializer(new Dictionary<string, Variable>());
            link.Serialize(serializer);
            Assert.Contains("fc/W", serializer.Target.Keys);
            Assert.Contains("fc/b", serializer.Target.Keys);
        }

        [Test]
        public void Dictからのデシリアライズ()
        {
            var link = new helper.models.VerySmallChain();
            var dict = new Dictionary<string, Variable>()
            {
                {"fc/W", new Variable(Matrix<float>.Build.DenseOfArray(new float[,] {{3, 2, 1}}))}
            };

            Helper.AssertMatrixNotAlmostEqual(
                link.fc._Params["W"].Value,
                dict["fc/W"].Value
            );

            var serializer = new DictionaryDeserializer(dict);
            link.Serialize(serializer);

            Helper.AssertMatrixAlmostEqual(
                link.fc._Params["W"].Value,
                dict["fc/W"].Value
            );
        }
    }
}
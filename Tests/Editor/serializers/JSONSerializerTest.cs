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
            UnityEngine.Debug.Log(serializer.Fetch());
//            Assert.Contains("fc/W", serializer.Target.Keys);
//            Assert.Contains("fc/b", serializer.Target.Keys);
        }

    }
}
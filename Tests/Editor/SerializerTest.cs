using NUnit.Framework;

namespace chainer
{
    public class SerializerTest
    {
        [NUnit.Framework.Test]
        public void シリアライズ()
        {
            var a = HDF5DotNet.H5.Open();
            UnityEngine.Debug.Log(HDF5DotNet.H5.Version);
        }
    }
}
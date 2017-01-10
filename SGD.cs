using System;

namespace chainer
{
    public class SGD
    {
        private float _lr;
        private Chain _chain;

        public SGD(float lr)
        {
            _lr = lr;
        }

        public void Setup(Chain chain)
        {
            _chain = chain;
        }

        public void ZeroGrads()
        {
            _chain.ClearGrads();
        }

        public void Update()
        {
            foreach (var kv in _chain.Params)
            {
                kv.Value.Value -= _lr * kv.Value.Grad;
            }
        }
    }
}
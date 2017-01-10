using System;

namespace chainer
{
    public class SGD
    {
        private float _lr;
        private Link _link;

        public SGD(float lr)
        {
            _lr = lr;
        }

        public void Setup(Link link)
        {
            _link = link;
        }

        public void ZeroGrads()
        {
            _link.ClearGrads();
        }

        public void Update()
        {
            foreach (var param in _link.GetParams())
            {
                param.Value -= _lr * param.Grad;
            }
        }
    }
}
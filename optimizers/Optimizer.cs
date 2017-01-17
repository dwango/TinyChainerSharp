using System;

namespace chainer.optimizers
{
    public abstract class Optimizer
    {
        protected Link _link = null;

        protected virtual void _Setup()
        {
        }

        public void Setup(Link link)
        {
            _link = link;
            _Setup();
        }

        public void ZeroGrads()
        {
            if (_link == null)
            {
                throw new NullReferenceException("optimizer should be setup first");
            }
            _link.ClearGrads();
        }

        public abstract void Update();
    }
}
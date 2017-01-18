using System.Collections.Generic;

namespace chainer.optimizers
{
    public class SGD : Optimizer
    {
        private readonly float _lr;
        private Dictionary<Variable, Variable> _states;

        public SGD(float lr)
        {
            _lr = lr;
        }


        protected override void _Update()
        {
            foreach (var param in _link.GetParams())
            {
                param.Value -= _lr * param.Grad;
            }
        }
    }
}
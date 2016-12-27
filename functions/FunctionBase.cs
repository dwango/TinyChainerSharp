using System.Collections.Generic;

namespace chainer.functions
{
    public abstract class FunctionBase
    {
        public IEnumerable<Variable> Inputs;
        public Variable Forward(IEnumerable<Variable> inputs)
        {
            Inputs = inputs;
            var result = _forward(inputs);
            result.SetCreator(this);
            return result;
        }

        protected abstract Variable _forward(IEnumerable<Variable> inputs);
    }
}
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

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

        public Variable Forward(Variable x0)
        {
            return Forward(new List<Variable>() {x0});
        }

        public Variable Forward(Variable x0, Variable x1)
        {
            return Forward(new List<Variable>() {x0, x1});
        }


        public IEnumerable<Matrix<float>> Backward(Matrix<float> gy)
        {
            return _backward(Inputs.Select(x => x.Value), gy);
        }

        protected abstract Variable _forward(IEnumerable<Variable> inputs);
        protected abstract IEnumerable<Matrix<float>> _backward(IEnumerable<Matrix<float>> inputs, Matrix<float> gy);
    }
}